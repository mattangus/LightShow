import subprocess
from glob import glob
import os
from tqdm import tqdm

#helper function for converting the MELD dataset to audio only

fmt_cmd = "ffmpeg -y -i {} -ab 160k -ac 2 -ar 44100 -vn {}"
all_files = glob("data/MELD.Raw/*/*.mp4")
vid_folder = "MELD.Raw"
audio_folder = "MELD.audio"

for in_file in tqdm(all_files):
    out_file = in_file.replace(vid_folder, audio_folder).replace(".mp4", ".wav")
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    
    command = fmt_cmd.format(in_file, out_file)
    # print(command)
    
    subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    