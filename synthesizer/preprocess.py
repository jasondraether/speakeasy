from multiprocessing.pool import Pool
from synthesizer import audio
from functools import partial
from itertools import chain
from encoder import inference as encoder
from encoder import inference_accent as encoder_accent
from pathlib import Path
from utils import logmmse
from tqdm import tqdm
import numpy as np
import librosa
from synthesizer.kaldi_interface import KaldiInterface

N_ACCENT = 7
N_SPEAKER = 28
ACCENT_DICT = {
    'BDL': [0, 0], # American
    'RMS': [0, 1],
    'SLT': [0, 2],
    'CLB': [0, 3],
    'ABA': [1, 4], # Arabic
    'SKA': [1, 5],
    'YBAA': [1, 6],
    'ZHAA': [1, 7],
    'BWC': [2, 8],# Mandarin
    'LXC': [2, 9],
    'NCC': [2, 10],
    'TXHC': [2, 11],
    'ASI': [3, 12], # Hindi
    'RRBI': [3, 13],
    'SVBI': [3, 14],
    'TNI': [3, 15],
    'HJK': [4, 16], # Korean
    'HKK': [4, 17],
    'YDCK': [4, 18],
    'YKWK': [4, 19],
    'EBVS': [5, 20], # Spanish
    'ERMS': [5, 21],
    'MBMPS': [5, 22],
    'NJS': [5, 23],
    'HQTV': [6, 24], # Vietnamese
    'PNV': [6, 25],
    'THV': [6, 26],
    'TLV': [6, 27],
}

def preprocess_l2arctic(dataset_root: Path, out_dir: Path, kaldi_dir:Path, n_processes: int,
                           skip_existing: bool, hparams, source_speaker: str):
    input_dirs = [dataset_root]
    print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
    assert all(input_dir.exists() for input_dir in input_dirs)

    # Create the output directories for each output file type
    out_dir.joinpath("mels").mkdir(exist_ok=True)
    out_dir.joinpath("audio").mkdir(exist_ok=True)
    out_dir.joinpath("ppgs").mkdir(exist_ok=True)


    # Create a metadata file
    metadata_fpath = out_dir.joinpath("train.txt")
    metadata_file = metadata_fpath.open("a" if skip_existing else "w", encoding="utf-8")

    # Preprocess the dataset
    ki = KaldiInterface(wav_scp=str(kaldi_dir.joinpath('wav.scp')), bnf_scp=str(kaldi_dir.joinpath('bnf', 'feats.scp')))
    speaker_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))
    func = partial(preprocess_speaker, out_dir=out_dir, ref_speaker_feat_interface=ki, skip_existing=skip_existing,
                   hparams=hparams, source_speaker=source_speaker)
    job = Pool(n_processes).imap(func, speaker_dirs)
    for speaker_metadata in tqdm(job, "L2ARCTIC", len(speaker_dirs), unit="speakers"):
        for metadatum in speaker_metadata:
            metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
    metadata_file.close()

    # Verify the contents of the metadata file
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
    mel_frames = sum([int(m[5]) for m in metadata])
    timesteps = sum([int(m[4]) for m in metadata])
    sample_rate = hparams.sample_rate
    hours = (timesteps / sample_rate) / 3600
    print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
          (len(metadata), mel_frames, timesteps, hours))
    print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
    print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
    # print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))


def preprocess_speaker(speaker_dir, out_dir: Path, ref_speaker_feat_interface, skip_existing: bool, hparams, source_speaker: str):
    metadata = []
    for wav_fpath in speaker_dir.glob("wav/*"):
        assert wav_fpath.exists()
        wav, _ = librosa.load(wav_fpath, hparams.sample_rate)
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
        wav, _ = librosa.effects.trim(wav, top_db=25)
        wav_cat_fname = '{}-{}'.format(speaker_dir.name, wav_fpath.stem)

        metadata.append(process_utterance(wav, ref_speaker_feat_interface, out_dir, wav_cat_fname, skip_existing, hparams, source_speaker))
    return [m for m in metadata if m is not None]


def process_utterance(wav: np.ndarray, ref_speaker_feat_interface: KaldiInterface, out_dir: Path, basename: str,
                      skip_existing: bool, hparams, source_speaker: str):
    ## FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - Both the audios and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.

    # Skip existing utterances if needed
    mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename)
    wav_fpath = out_dir.joinpath("audio", "audio-%s.npy" % basename)
    ppg_fpath = out_dir.joinpath("ppgs", "ppg-%s.npy" % basename)
    if skip_existing and mel_fpath.exists() and wav_fpath.exists():
        return None

    # Skip utterances that are too short
    if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
        return None

    # Compute the mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    # Skip utterances that are too long
    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None

    # Compute ppg
    ppg = ref_speaker_feat_interface.get_feature(source_speaker.upper()+'_'+basename.split('-')[-1], 'bnf')
    # ppg_frames = ppg.shape[0]

    # Sometimes ppg can be 1 frame longer than mel
    # min_frames = min(mel_frames, ppg_frames)
    # mel_spectrogram = mel_spectrogram[:, :min_frames]
    # ppg = ppg[:min_frames, :]

    # Write the spectrogram, embed, ppg and audio to disk
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)
    np.save(ppg_fpath, ppg, allow_pickle=False)

    # Return a tuple describing this training example
    return wav_fpath.name, mel_fpath.name, ppg_fpath.name, "embed-%s.npy" % basename, len(wav), mel_frames


def embed_utterance_dvec(fpath, encoder_model_fpath, encoder_accent_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    if not encoder_accent.is_loaded():
        encoder_accent.load_model(encoder_accent_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath, embed_fpath = fpath
    wav = np.load(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed_speaker = encoder.embed_utterance(wav)
    embed_accent = encoder_accent.embed_utterance(wav)
    embed = np.concatenate((embed_accent, embed_speaker))
    np.save(embed_fpath, embed, allow_pickle=False)


def create_dvec_embeddings(synthesizer_root: Path, encoder_model_fpath: Path, encoder_accent_model_fpath: Path, n_processes: int):
    wav_dir = synthesizer_root.joinpath("audio")
    metadata_fpath = synthesizer_root.joinpath("train.txt")
    assert wav_dir.exists() and metadata_fpath.exists()
    embed_dir = synthesizer_root.joinpath("embeds")
    embed_dir.mkdir(exist_ok=True)

    # Gather the input wave filepath and the target output embed filepath
    with metadata_fpath.open("r") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
        fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[3])) for m in metadata]

    # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
    # Embed the utterances in separate threads
    func = partial(embed_utterance_dvec, encoder_model_fpath=encoder_model_fpath, encoder_accent_model_fpath=encoder_accent_model_fpath)
    job = Pool(n_processes).imap(func, fpaths)
    list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))