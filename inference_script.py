import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from synthesizer.inference import Synthesizer
from synthesizer.kaldi_interface import KaldiInterface
from encoder import inference as encoder
from vocoder import inference as vocoder
import numpy as np
import librosa
from synthesizer.hparams import hparams
from scipy.io import wavfile
from speechbrain.pretrained import SpeakerRecognition
import tempfile
from random import shuffle
import pandas as pd
import sys
import datetime

# Paths to dataset and models. Also initialize temp directory.
dataset_dir = "/home/jason/workspace/datasets/daic/clean/train/"
encoder_speaker_weights = "/home/jason/workspace/speakeasy/models/encoder/saved_models/pretrained.pt"
vocoder_weights = "/home/jason/workspace/speakeasy/models/vocoder/saved_models/pretrained/pretrained.pt"
syn_dir = "/home/jason/workspace/speakeasy/models/tacotron_pretrained_l2arctic/tacotron_model.ckpt-204001"
results_path = f"/home/jason/workspace/speakeasy/results/{datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}-results.csv"
tmp_dir = tempfile.gettempdir()

# Load all models
print('Loading encoder...')
encoder.load_model(encoder_speaker_weights)
print('Loading synthesizer...')
synthesizer = Synthesizer(syn_dir)
print('Loading vocoder...')
vocoder.load_model(vocoder_weights)
print('Loading verification...')
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# Helper function to synthesize new audio
def synthesize(bnf, embed):
    spec = synthesizer.synthesize_spectrograms([bnf], [embed])[0]
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    return generated_wav

# Helper function to get speaker embedding
def generate_speaker_embed(tgt_utterance_path):
    wav, _ = librosa.load(tgt_utterance_path, sr=hparams.sample_rate)
    wav = encoder.preprocess_wav(wav)
    embed_speaker = encoder.embed_utterance(wav)

    return embed_speaker

# The source speaker is the individual whose linguistic content we're extracting
source_speaker = 'speaker_350'
utterance_no = '50'
source_dir = os.path.join(dataset_dir, source_speaker)
kaldi_dir = os.path.join(source_dir, 'kaldi/')
source_utterance = os.path.join(source_dir, 'wav/', f'{utterance_no}.wav')

# Initialize kaldi interface to source speaker
print(f'KALDI DIR IS {kaldi_dir}')
ki = KaldiInterface(
    wav_scp=os.path.join(kaldi_dir, 'wav.scp'),
    bnf_scp=os.path.join(kaldi_dir, 'bnf/feats.scp')
)
# Get bnf for source utterance
bnf = ki.get_feature(f'{source_speaker}_{utterance_no}', 'bnf')

# Read speaker directories for other speakers, exclude source speaker
speaker_dirs = [os.path.join(dataset_dir, s_name, 'wav/') for s_name in os.listdir(dataset_dir) if s_name != source_speaker]

# Track results
results = {
    'speaker': [], 
    'source_file': [], 
    'target_file': [], 
    'score': [], 
    'prediction': [], 
    'expected': []
}

# Loop through speakers and take turns with each speaker being the target speaker
# We convert the utterance of the source speaker to sound like the target speaker
# We then check if the speaker verification system with this converted utterance
# The verification system should THINK the new utterance came from the target speaker
for s in speaker_dirs[:2]:
    print(f"Reading and shuffling utterances from {s}") 
    utts = os.listdir(s)
    shuffle(utts)

    n_utts = len(utts)

    if n_utts < 10:
        raise ValueError(f'Insufficient utterances in directory {s}')

    print(f"Found {n_utts} utterances")

    pool = utts 
    test_pool = utts

    for i, u in enumerate(pool[:2]):
        print(f"Running {i+1}/{n_utts}")

        utt_path = os.path.join(s, u)
        utt_emb = generate_speaker_embed(utt_path)

        utt_synth = synthesize(bnf, utt_emb)

        output_path = os.path.join(tmp_dir, f'temp_conv.wav')
        wavfile.write(output_path, hparams.sample_rate, utt_synth)

        # Here we compare the converted utterance to that same utterance of the
        # source speaker. The verification system should think the utterances came from
        # a DIFFERENT person
        score, prediction = verification.verify_files(
            output_path,
            source_utterance
        ) # Expect false

        results['speaker'].append(s)
        results['source_file'].append(source_utterance)
        results['target_file'].append(utt_path)
        results['score'].append(float(score[0]))
        results['prediction'].append(bool(prediction[0]))
        results['expected'].append(False)

        for t_u in test_pool[2:12]:
            # Here we compare the converted utterance to other utterances of the target
            # speaker. The verification system should think the utterances came from the same person
            utt_path = os.path.join(s, t_u)


            score, prediction = verification.verify_files(
                output_path,
                utt_path
            ) # Expect true

            results['speaker'].append(s)
            results['source_file'].append(source_utterance)
            results['target_file'].append(utt_path)
            results['score'].append(float(score[0]))
            results['prediction'].append(bool(prediction[0]))
            results['expected'].append(True)

df = pd.DataFrame(results)
df.to_csv(results_path, header=True, index=False)


