# Speaker Anonymization using Voice Conversion
This implementation uses voice conversion forked from [here](https://github.com/warisqr007/voice-conversion), which was originally adapted from [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning) to perform speaker anonymization. We define a source speaker as the individual who we wish to anonymize but preserve their linguistic information (i.e., the words they speak) as well as the emotional affect (e.g., sadness, happiness, anger) contained in their original utterance.

## Things to know 
The .wav files in the DAIC dataset are 16-bit PCM (i.e., 16-bit integers) monosampled at 16000 Hz. The output of the Tacotron synthesizer is 32-bit float, but is converted back into 16-bit PCM for you in the `inference_script.py` code.

Also, `speechbrain` (the speaker verification system) will make symbolic links to any two files you compare using it. So, if the directory gets filled with a bunch of .wav files, they're symbolic and can be deleted.

## Setup (for Ubuntu 20.04 LTS)
**Python** 

1. Install Python 3.8.
2. Setup a virtual environment `python -m venv env` and activate the environment `source env/bin/activate`.
3. Install requirements `pip install -r requirements.txt`.
4. Configure CUDA to work with PyTorch and Tensorflow. This varies based on your GPU and your setup, so this step is left up to you. CUDA isn't necessary, but the code will require some modifications if CUDA isn't enabled.
**Kaldi**

5. Clone Kaldi repository `git clone https://github.com/kaldi-asr.git kaldi --origin upstream` and enter repo folder `cd kaldi`.
6. Follow `INSTALL.md` instructions in the root folder (go to `tools/`, follow `INSTALL.md` instructions there, then go to `src/` and follow `INSTALL.md` instructions there).
**Models**

7. Install all of the required models
7.1 Encoder, Vocoder, Synthesizer, and Acoustic models can be downloaded [here](https://drive.google.com/file/d/1HdHqIk3ij2h9m5NqfgWK19OJqEGAgoJv/view?usp=sharing). The Encoder, Vocoder, and Synthesizer are originally from [LibriSpeech](https://www.openslr.org/12), and the Synthesizer is from [ARCTIC](http://www.festvox.org/cmu_arctic/) and [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/). 
7.2 Mozilla Deep Speech can be downloaded [here](https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm) for the model and [here](https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer) for the scorer.

**Dataset** 

8. The dataset used is the DAIC dataset. You must request access for it first. See [here](https://dcapswoz.ict.usc.edu/) for more information.
9. Once you have the dataset, you can use the `format-dataset.py` script to format it. Change paths in the script as needed.

**Setting Paths**

10. There are a few paths you have to set in the code.
10.1 In `kaldi_scripts/extract_features_kaldi.sh`, set `KALDI_ROOT` equal to the path of your Kaldi installation (e.g., `KALDI_ROOT=/home/user/path/to/kaldi/`), set `PRETRAIN_ROOT` equal to the path of the acaoustic model you downloaded in step 7.1 (e.g., `PRETRAIN_ROOT=/home/user/path/to/0013_librispeech_v1/`, and set `SCRIPT_ROOT` equal to the path of the `extract_features_kaldi.sh` script (e.g., `SCRIPT_ROOT=/home/user/path/to/kaldi_scripts/`)
10.2 In `inference_script.py`, set `dataset_dir` equal to the path of your formatted dataset (a folder of a list of speakers that each have a `wav/` and `kaldi/` folder in them, e.g., `/path/to/daic/speaker_350/wav/1.wav`. Set `encoder_speaker_weights` to the path of the encoder (should end in `pretrained.pt`. Set `vocoder_weights` equal to the path of the vocoder (should end in `pretrained.pt`). Set `syn_dir` equal to the path of your synthesizer (mine was `[...]/tacotron_pretrained_l2arctic/tacotron_model.ckpt-204001`), set `stt_model_path` and `stt_scorer_path` equal to the `.pbmm` and `.scorer` files you downloaded from Mozilla, and make sure `results_path` points to a valid folder (you can leave the filename generated based on datetime if you want, or change it).

**Formatting and Preprocessing**
11. To format the dataset for use with `inference_script.py` and `extract_features_kaldi.sh`, run the `format_dataset.py` script, setting the `source_dir` variable to the path which has the folder containing the utterances for all speakers (e.g., `daic/`, where `speaker_350_uttr5.wav`, `speaker_371_uttr10.wav`, etc. is). Set the `output_dir` variable to the folder where you'd like to output the formatted dataset. The folder needs to exist first.

12. To extract features using `extract_features_kaldi.sh`, after you've formatted the dataset, make sure `extract_features_kaldi.sh` is an executable (`chmod +x extract_features_kaldi.sh` if not), and run it as `./extract_features_kaldi.sh /absolute/path/to/speaker/folder/`. You'll likely need to run this for multiple speakers, so you can do something like `for s in /output_folder/*; do ./extract_features_kaldi.sh /output_folder/${s}; done;` in the terminal. A folder called `kaldi/` will be added to each speaker in `/output_folder/`.

**Running**

13. Once everything is setup, simply run `python inference_script.py` to start generating results. The results will end up in the folder according to `results_path`.

**Other Links**
[Resemblyzer](https://github.com/resemble-ai/Resemblyzer)
[Google GE2E](https://arxiv.org/pdf/1710.10467.pdf)
[Accentron](https://psi.engr.tamu.edu/wp-content/uploads/2021/10/1-s2.0-S0885230821001029-main.pdf)

**(Old, for reference) Training**

* Use Kaldi to extract BNF for the reference L1 speaker
```
./kaldi_scripts/extract_features_kaldi.sh /path/to/L2-ARCTIC/BDL
```
* Preprocessing
```
python synthesizer_preprocess_audio.py /path/to/L2-ARCTIC BDL /path/to/L2-ARCTIC/BDL/kaldi --out_dir=your_preprocess_output_dir
python synthesizer_preprocess_embeds.py your_preprocess_output_dir
```
* Training
```
python synthesizer_train.py Accetron_train your_preprocess_output_dir
```
