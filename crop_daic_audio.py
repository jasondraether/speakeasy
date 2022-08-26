# Standard libraries
import re
import os

# 3P libraries
import pandas as pd
import numpy as np
from pydub import AudioSegment
import soundfile
from scipy.io import wavfile

# Given a wavfile and a start/end timestamp in seconds, grab segment between timestamps
def segment_audio(audio, start_ts: float, end_ts: float):
    start_ts_m = start_ts * 1000 # Convert timestamp (s) to timestamp (ms)
    end_ts_m = end_ts * 1000 
    segment = audio[start_ts_m:end_ts_m] # Grab segment
    return segment

# Cleanup transcript file from DAIC
def preprocess_transcript(transcript_path: str):
    df = pd.read_csv(transcript_path, header=0, sep='\t')
    # We only care about participants (also strip the <synch> row)
    df = df[(df['speaker'] == 'Participant') & (df['value'] != '<synch>')]
    # Clean up value column (strip <> comments)
    df['value'] = df['value'].apply(lambda x: re.sub('<.*?>', '', x))
    # Remove rows with no words
    df = df[df['value'] != '']
    return df

# Formatting for speaker folders
def speaker_id_to_dir(speaker_id: int):
    return f'speaker_{speaker_id}/'

# Formatting for utterance filenames
def utt_no_to_fname(utt_no: int):
    return f'{utt_no}.wav'

# Process a transcript + wavfile for a given speaker, return segments
def process_file(transcript_path: str, audio_path: str, speaker_id: int, output_dir: str):
    # Create speaker folder if it doesn't exist
    speaker_dir = os.path.join(output_dir, speaker_id_to_folder(speaker_id))
    if not os.path.exists(speaker_dir):
        print(f'Folder {speaker_dir} does not existing. Creating it in {output_dir}.')
        os.mkdir(speaker_dir)

    # Get audio and transcript (cleanup)
    df = preprocess_transcript(transcript_path)
    audio = AudioSegment.from_wav(audio_path)

    # Audio metafile (segment_output_path -> transcript)
    meta = {}
    meta_output_path = os.path.join(speaker_dir, 'meta.json')

    # Segment each utterance
    for utt_no, (start_time, end_time, text) in enumerate(zip(df['start_time'], df['end_time'], df['value'])):
        segment = segment_audio(audio, start_time, end_time)
        segment_output_path = os.path.join(speaker_dir, utt_no_to_fname(utt_no))
        print(f'Creating segment file {segment_output_path}.')
        soundfile.write(segment_output_path, segment.get_array_of_samples(), 16000, subtype='PCM_16')
        # Track transcript for file
        meta[utt_no_to_fname(utt_no)] = text

    # Write metafile to speaker folder
    print(f'Writing meta file to {meta_output_path}.')
    with open(meta_output_path, 'r') as f:
        json.dump(meta, f, indent=4, sort_keys=True)

    print(f'Done processing speaker {speaker_id}.')



    
# Main starts here

# Setting directories
# Where you want the segments to be written to
output_dir = '/home/jason/workspace/datasets/daic-new/'
# Source dataset. The folder specified here should have a list of folders
# in the form of {speaker_id}_P/ (e.g., '350_P/' for speaker 350). Inside
# each speaker folder should be the transcript, named '{speaker_id}_TRANSCRIPT.csv', and
# the audio file, named '{speaker_id}_AUDIO.wav'
source_dir = '/path/to/dataset/source/'

# Create output_dir if it doesn't exist
if not os.path.exists(output_dir):
    print(f'Directory {output_dir} does not exist. Creating.')
    os.mkdir(output_dir)




