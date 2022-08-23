# Standard libraries
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tempfile
import random
from random import shuffle
import sys
import datetime
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

# Set seed
random.seed(123)

# Maintain cache
cache = {}

# Helper function to synthesize new audio
def synthesize(bnf, embed):
    # Generate spectrogram from bottleneck feature and speaker embedding
    spec = synthesizer.synthesize_spectrograms([bnf], [embed])[0]
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    generated_wav *= 32767

    return generated_wav.astype(np.int16)  

# Helper function to get speaker embedding
def generate_speaker_embed(tgt_utterance_path):
    # Load wav for utterance, preprocess, and grab embedding
    wav, _ = librosa.load(tgt_utterance_path, sr=hparams.sample_rate)
    wav = encoder.preprocess_wav(wav)
    embed_speaker = encoder.embed_utterance(wav)

    return embed_speaker

def to_row(path_1, path_2, score, pred, expected, test_type, source_speaker, target_speaker, word_error_rate):
    return {
        'source_path': path_1,
        'compared_path': path_2,
        'score': score,
        'prediction': pred,
        'expected': expected,
        'type': test_type,
        'source_speaker': source_speaker,
        'target_speaker': target_speaker,
        'wer': word_error_rate
    }



def verify_converted(
    conv_path, 
    test_path, 
    source_speaker, 
    target_speaker,
    expected,
    label
):
    score, prediction = verification.verify_files(
        conv_path,
        test_path
    )

    # If we're comparing converted with source, compute WER
    if label == 'CONV-SOURCE_ORIG':
        sr1, conv_sig = wavfile.read(conv_path)
        sr2, test_sig = wavfile.read(test_path)
        assert sr1 == sr2

        conv_text = ds.stt(conv_sig)
        test_text = ds.stt(test_sig)

        if len(test_text) == 0:
            error = np.nan
        else:
            error = wer(test_text, conv_text)
    else:
        error = 0.0


    return to_row(
        conv_path,
        test_path,
        float(score[0]),
        bool(prediction[0]),
        expected,
        label,
        source_speaker,
        target_speaker,
        error
    )
    

# Helper function to perform the process for two speakers. Returns a dict
# for the results
def process_batch(bnf, target_path, target_pool, target_speaker, source_path, source_pool, source_speaker):
    print(f'Computing results for source speaker {source_speaker} and target_speaker {target_speaker}.')

    if target_path in cache:
        utt_emb = cache[target_path]
        print('Cache hit')
    else:
        utt_emb = generate_speaker_embed(target_path)
        cache[target_path] = utt_emb
    utt_synth = synthesize(bnf, utt_emb)

    temp_path = os.path.join(tmp_dir, f'converted_wav_tmp.wav')
    wavfile.write(temp_path, hparams.sample_rate, utt_synth)

    batch_results = []

    # Compare converted utterance with original target utterance
    batch_results.append(verify_converted(temp_path, target_path, source_speaker, target_speaker, True, 'CONV-TARGET_ORIG'))
    # Compare converted utterance with original source utterance (WER is also computed here)
    batch_results.append(verify_converted(temp_path, source_path, source_speaker, target_speaker, False, 'CONV-SOURCE_ORIG'))
    # Compare converted utterance with target utterances from test pool
    for f in target_pool:
        batch_results.append(verify_converted(temp_path, f, source_speaker, target_speaker, True, 'CONV-TARGET_POOL'))
    # Compare converted utterance with source utterances from test pool
    for f in source_pool:
        batch_results.append(verify_converted(temp_path, f, source_speaker, target_speaker, True, 'CONV-SOURCE_POOL'))

    return batch_results

# Number of utterances to use, globally
DATUM_SIZE = 8

# Paths to dataset and models. Also initialize temp directory.
dataset_dir = "/home/jason/workspace/datasets/daic/clean/train/"
encoder_speaker_weights = "/home/jason/workspace/speakeasy/models/encoder/saved_models/pretrained.pt"
vocoder_weights = "/home/jason/workspace/speakeasy/models/vocoder/saved_models/pretrained/pretrained.pt"
syn_dir = "/home/jason/workspace/speakeasy/models/tacotron_pretrained_l2arctic/tacotron_model.ckpt-204001"
results_path = f"/home/jason/workspace/results/{datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}-results.csv"
stt_model_path = "/home/jason/workspace/speakeasy/models/mozilla/model.pbmm"
stt_scorer_path = "/home/jason/workspace/speakeasy/models/mozilla/model.scorer"
tmp_dir = tempfile.gettempdir()

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
print('========================== Loading stt ====================================')
ds = Model(stt_model_path)
ds.enableExternalScorer(stt_scorer_path)

# Get number of speakers in dataset, shuffle them
speakers = os.listdir(dataset_dir)
n_speakers = len(speakers)
shuffle(speakers)

# Half of the speakers are source speakers (anonymize), 
# the other half are target speakers (make it sound like them)
source_speakers = speakers[:n_speakers//2]
target_speakers = speakers[n_speakers//2:]

results = []

try:
    # Go through each speaker in the source speaker pool
    for source_speaker in source_speakers:
        print(f'===== On source speaker {source_speaker} =====')
        source_dir = os.path.join(dataset_dir, source_speaker)
        kaldi_dir = os.path.join(source_dir, 'kaldi/')
        wav_dir = os.path.join(source_dir, 'wav/')
        # Pull source speaker utterances
        source_utterances = os.listdir(wav_dir)
        if len(source_utterances) < DATUM_SIZE:
            print(f'Not enough utterances for source speaker {source_speaker}. Check dataset.')
            continue

        shuffle(source_utterances)
        # Only select a subset of source utterances. Use half for converting and other half
        # for testing
        source_utterances = source_utterances[:DATUM_SIZE]
        batch_source_utterances = source_utterances[:DATUM_SIZE//2]
        test_source_utterances = source_utterances[DATUM_SIZE//2:]
        test_source_utterance_paths = [os.path.join(wav_dir, utt) for utt in test_source_utterances]

        source_utterance_nums = [os.path.splitext(s_u)[0] for s_u in batch_source_utterances]

        ki = KaldiInterface(
            wav_scp=os.path.join(kaldi_dir, 'wav.scp'),
            bnf_scp=os.path.join(kaldi_dir, 'bnf/feats.scp')
        )

        # Go through each source utterance in conversion pool. Convert with
        # each target speaker
        for i, source_utterance_num in enumerate(source_utterance_nums):
            print(f'===== Source Utterance {i+1}/{DATUM_SIZE//2} =====')
            bnf = ki.get_feature(f'{source_speaker}_{source_utterance_num}', 'bnf')
            if bnf is None:
                print(f'Error extracting bnf for utterance {source_utterance_num} for source speaker {source_speaker}. Skipping')
                continue
            source_utterance_path = os.path.join(wav_dir, f'{source_utterance_num}.wav')

            for target_speaker in target_speakers[:5]:
                print(f'===== On target speaker {target_speaker} =====')
                target_dir = os.path.join(dataset_dir, target_speaker, 'wav/')

                target_utterances = os.listdir(target_dir)
                if len(target_utterances) < DATUM_SIZE:
                    print(f'Not enough utterances for target speaker {target_speaker}. Check dataset.')
                    continue

                shuffle(target_utterances)
                target_utterances = target_utterances[:DATUM_SIZE]
                batch_target_utterances = target_utterances[:DATUM_SIZE//2]
                test_target_utterances = target_utterances[DATUM_SIZE//2:]
                test_target_utterance_paths = [os.path.join(target_dir, utt) for utt in test_target_utterances]

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
                        results += sample_results
                    except Exception as e:
                        print(f'Error with batch, skipping: {e}')
                    

except KeyboardInterrupt:
    print('Exiting early.')
except Exception as e:
    print(e)
    print('Error occurred during loop. Exiting early.')

df = pd.DataFrame(results)
df.to_csv(results_path, header=True, index=False)
print(f'Results written to {results_path} with {len(df.index)} rows.')


