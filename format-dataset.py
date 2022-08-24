import os
import shutil
import glob
import re

sessions = range(300, 493)
source_dir = 'raw/dev/'
output_dir = 'cleaned/dev/'

pattern_string = re.compile(r"spk_[0-9]+_uttr([0-9]+)")

for s in sessions:
    target_pattern = f'spk_{s}_uttr*.wav'
    print(target_pattern)
    files = glob.glob(os.path.join(source_dir, target_pattern))
    
    output_session_dir = os.path.join(output_dir, f'speaker_{s}/', 'wav/')
    
    if len(files) > 0:
        if not os.path.exists(output_session_dir):
            os.makedirs(output_session_dir)
        
        for f in files:
            uids = pattern_string.findall(f)
            if len(uids) != 1:
                raise ValueError(f"Invalid file name {f}")
            output_path = os.path.join(output_session_dir, f'{uids[0]}.wav')


            shutil.copy(f, output_path) 
            
    
