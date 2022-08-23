# Copyright 2022 Jason Raether

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# About
"""
This script is used for gathering batch results for voice anonymization.
It currently loops through source and target speakers, converts the source
speaker to sound like the target speaker (while keeping the linguistic content
of the source speaker), and tests speaker verification and word error rate (WER).
"""

# Standard libraries
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tempfile
import random
from random import shuffle
import sys
import datetime
import traceback
# 3P libraries
import torch
import numpy as np
import librosa
from scipy.io import wavfile
import pandas as pd
from speechbrain.pretrained import SpeakerRecognition
from deepspeech import Model
from jiwer import wer
# Local libraries
from synthesizer.hparams import hparams
from synthesizer.inference import Synthesizer
from synthesizer.kaldi_interface import KaldiInterface
from encoder import inference as encoder
from vocoder import inference as vocoder

# Constants
DATUM_SIZE = 8 # The total number of utterances to use for testing + pool
# E.g., if DATUM_SIZE = 8, then 4 utterances are used for source and target speakers and 4 are for their respective pools
dataset_dir = "/home/jason/workspace/datasets/daic/clean/train/" # Path to DAIC datasest
encoder_speaker_weights = "/home/jason/workspace/speakeasy/models/encoder/saved_models/pretrained.pt" # Path to encoder .pt file
vocoder_weights = "/home/jason/workspace/speakeasy/models/vocoder/saved_models/pretrained/pretrained.pt" # Path to vocoder .pt file
syn_dir = "/home/jason/workspace/speakeasy/models/tacotron_pretrained_l2arctic/tacotron_model.ckpt-204001" # Path to tacotron .ckpt file
results_path = f"/home/jason/workspace/results/{datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}-results.csv" # Output path for results (filename based on datetime)
stt_model_path = "/home/jason/workspace/speakeasy/models/mozilla/model.pbmm" # Path to .pbmm for deepspeech speech2text
stt_scorer_path = "/home/jason/workspace/speakeasy/models/mozilla/model.scorer" # Path to .scorer for deepspeech speech2text
tmp_dir = tempfile.gettempdir() # Temporary directory path for writing temp files

# Set seed (doesn't matter what it is, just that its consistent)
random.seed(123)

# Helper function to synthesize new audio
def synthesize(bnf, embed):
    # Generate spectrogram from bottleneck feature (BNF) and speaker embedding
    spec = synthesizer.synthesize_spectrograms([bnf], [embed])[0]
    # Generate waveform from spectrogram
    generated_wav = vocoder.infer_waveform(spec)
    # Pad waveform
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    # Convert from float32 to range of int16 (generated_wav is float32)
    generated_wav *= 32767 
    # Convert type to int16
    return generated_wav.astype(np.int16)  

# Helper function to get speaker embedding
def generate_speaker_embed(tgt_utterance_path: str):
    # Load wav for utterance, preprocess, and grab embedding
    wav, _ = librosa.load(tgt_utterance_path, sr=hparams.sample_rate)
    wav = encoder.preprocess_wav(wav)
    embed_speaker = encoder.embed_utterance(wav)

    return embed_speaker

# Helper function to generate a row of data and compute metrics
# for two given paths
def verify_converted(
    conv_path: str, 
    test_path: str, 
    source_speaker: str, 
    target_speaker: str,
    expected: bool,
    label: str
):
    # Verify between the converted path and the path being tested
    score, prediction = verification.verify_files(
        conv_path,
        test_path
    )

    # If we're comparing converted with original source (CONV-SOURCE_ORIG), compute WER
    # WER will tell us how much the converter screwed up the linguistic content
    if label == 'CONV-SOURCE_ORIG':
        sr1, conv_sig = wavfile.read(conv_path)
        sr2, test_sig = wavfile.read(test_path)
        assert sr1 == sr2

        conv_text = ds.stt(conv_sig)
        test_text = ds.stt(test_sig)

        # Sometimes stt thinks there are no words in which case this test is undefined, so nan for now
        if len(test_text) == 0:
            error = np.nan
        else:
            error = wer(test_text, conv_text)
    else:
        error = 0.0 # Set arbitrarily (we won't look at this in the analysis)

    # Return a row of data
    return {
        'source_path': conv_path,
        'compared_path': test_path,
        'score': float(score[0]),
        'prediction': bool(prediction[0]),
        'expected': expected,
        'type': label,
        'source_speaker': source_speaker,
        'target_speaker': target_speaker,
        'wer': error
    }
    

# Helper function to perform the process for two speakers. Returns a dict
# for the results
def process_batch(bnf, target_path, target_pool, target_speaker, source_path, source_pool, source_speaker):
    print(f'Computing results for source speaker {source_speaker} and target_speaker {target_speaker}.')
    # Generate embeddings and synthesize utterance
    utt_emb = generate_speaker_embed(target_path)
    utt_synth = synthesize(bnf, utt_emb)
    # Write to a temporary file into temporary dir
    temp_path = os.path.join(tmp_dir, f'converted_wav_tmp.wav')
    wavfile.write(temp_path, hparams.sample_rate, utt_synth)
    # Store results
    batch_results = []
    # Compare converted utterance with original target utterance (CONV-TARGET_ORIG)
    batch_results.append(verify_converted(temp_path, target_path, source_speaker, target_speaker, True, 'CONV-TARGET_ORIG'))
    # Compare converted utterance with original source utterance (WER is also computed here) (CONV-SOURCE_ORIG)
    batch_results.append(verify_converted(temp_path, source_path, source_speaker, target_speaker, False, 'CONV-SOURCE_ORIG'))
    # Compare converted utterance with target utterances from test pool (CONV-TARGET_POOL)
    for f in target_pool:
        batch_results.append(verify_converted(temp_path, f, source_speaker, target_speaker, True, 'CONV-TARGET_POOL'))
    # Compare converted utterance with source utterances from test pool (CONV-SOURCE_POOL)
    for f in source_pool:
        batch_results.append(verify_converted(temp_path, f, source_speaker, target_speaker, True, 'CONV-SOURCE_POOL'))

    return batch_results


# Load all models
print('========================== Loading encoder ==========================')
encoder.load_model(encoder_speaker_weights, device=0)
print('========================== Loading synthesizer ==========================')
synthesizer = Synthesizer(syn_dir)
print('========================== Loading vocoder ==========================')
vocoder.load_model(vocoder_weights)
print('========================== Loading verification ==========================')
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)
print('========================== Loading speech2text ====================================')
ds = Model(stt_model_path)
ds.enableExternalScorer(stt_scorer_path)
print('========================== All models loaded =================================')

# Read dataset to get list of speakers. Count them and shuffle them.
speakers = os.listdir(dataset_dir)
n_speakers = len(speakers)
shuffle(speakers)

# Half of the speakers are source speakers (anonymize their voice but keep their linguistic content), 
# the other half are target speakers (make the source sound like them)
source_speakers = speakers[:n_speakers//2]
target_speakers = speakers[n_speakers//2:]

# Store all of the results to be written to a file at the end
results = []

# Wrap this in a try-except
# * If you want to interrupt w/ CTRL+C, it will save any results you currently have
# * If you encounter an exception, it will show the exception and save any results you currently have
try:
    # Go through each speaker in the source speaker group
    # Overview of the process:
    # Each source speaker and target speaker has test utterances and pool utterances
    # A test source speaker utterance and test target speaker utterance are used
    # for utterance conversion, while pool utterances are used for verifying the converted utterance
    for source_speaker in source_speakers:
        print(f'===== On source speaker {source_speaker} =====')
        # Grab required directories for kaldi interface
        source_dir = os.path.join(dataset_dir, source_speaker)
        kaldi_dir = os.path.join(source_dir, 'kaldi/')
        wav_dir = os.path.join(source_dir, 'wav/')
        # Read list of source speaker utterances
        source_utterances = os.listdir(wav_dir)
        if len(source_utterances) < DATUM_SIZE:
            print(f'Not enough utterances for source speaker {source_speaker}. Check dataset.')
            continue
        # Shuffle the utterances for random
        shuffle(source_utterances)
        # Only select a subset of source utterances. Use half for converting and other half for testing
        source_utterances = source_utterances[:DATUM_SIZE]
        batch_source_utterances = source_utterances[:DATUM_SIZE//2]
        test_source_utterances = source_utterances[DATUM_SIZE//2:]
        # Generate the full paths to the source utterances we're using for the source utterance pool
        test_source_utterance_paths = [os.path.join(wav_dir, utt) for utt in test_source_utterances]
        # Generate the numbers for the source utterances we're using for converting
        source_utterance_nums = [os.path.splitext(s_u)[0] for s_u in batch_source_utterances]
        # Instantiate the kaldi interface to read BNF vectors
        ki = KaldiInterface(
            wav_scp=os.path.join(kaldi_dir, 'wav.scp'),
            bnf_scp=os.path.join(kaldi_dir, 'bnf/feats.scp')
        )
        # Go through each source test utterance. Convert with each target speaker and their test utterances
        for i, source_utterance_num in enumerate(source_utterance_nums):
            print(f'===== Source Utterance {i+1}/{DATUM_SIZE//2} =====')
            # Get BNF for this source utterance
            bnf = ki.get_feature(f'{source_speaker}_{source_utterance_num}', 'bnf')
            if bnf is None:
                print(f'Error extracting bnf for utterance {source_utterance_num} for source speaker {source_speaker}. Skipping')
                continue
            source_utterance_path = os.path.join(wav_dir, f'{source_utterance_num}.wav')

            # Feel free to truncate the target speakers since the script may take too long (I have it set to 5 here)
            for target_speaker in target_speakers[:5]:
                print(f'===== On target speaker {target_speaker} =====')
                target_dir = os.path.join(dataset_dir, target_speaker, 'wav/')
                # Read the target speaker utterances
                target_utterances = os.listdir(target_dir)
                if len(target_utterances) < DATUM_SIZE:
                    print(f'Not enough utterances for target speaker {target_speaker}. Check dataset.')
                    continue
                # Shuffle utterance, split into test and pool utterances (same as before w/ source)
                shuffle(target_utterances)
                target_utterances = target_utterances[:DATUM_SIZE]
                batch_target_utterances = target_utterances[:DATUM_SIZE//2]
                test_target_utterances = target_utterances[DATUM_SIZE//2:]
                test_target_utterance_paths = [os.path.join(target_dir, utt) for utt in test_target_utterances]
                # Now, go through source test utterances and target test utterances and convert. Then
                # check the conversion against the source pool utterances and target pool utterances
                for j, target_utterance in enumerate(batch_target_utterances):
                    print(f'===== Target Utterance {j+1}/{DATUM_SIZE//2} =====')
                    target_utterance_path = os.path.join(target_dir, target_utterance)
                    try:
                        sample_results = process_batch(
                            bnf, 
                            target_utterance_path, 
                            test_target_utterance_paths,
                            target_speaker,
                            source_utterance_path,
                            test_source_utterance_paths,
                            source_speaker
                        )
                        # Concat results to global results
                        results += sample_results 
                    except Exception as e:
                        print(f'Error with batch, skipping: {e}')
                    

except KeyboardInterrupt:
    print('Exiting early.')
except Exception as e:
    print(traceback.format_exc())
    print(f'Error occurred during loop. Exiting early. Error was: {e}\nSee traceback for more info.')

df = pd.DataFrame(results)
df.to_csv(results_path, header=True, index=False)
print(f'Results written to {results_path} with {len(df.index)} rows.')


