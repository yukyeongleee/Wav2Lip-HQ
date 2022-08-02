import os
import subprocess
import numpy as np

from tqdm import tqdm

import audio

parser = argparse.ArgumentParser()

parser.add_argument('--data_root', 	type=str, default="/data1/HDTF/", help='');
parser.add_argument('--wav_root', type=str, default="/home/leee/data/HDTF/audio-16k/", help='');
parser.add_argument('--mel_root', type=str, default="/home/leee/data/HDTF/mel/", help='');
# parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate');
args = parser.parse_args();


def main(args):
    # data_root = "/data1/HDTF/"
    # wav_root = "/home/leee/data/HDTF/audio-16k/"
    # mel_root = "/home/leee/data/HDTF/mel/"
    os.makedirs(wav_root, exist_ok=True)
    os.makedirs(mel_root, exist_ok=True)

    audio_list = [f for f in os.listdir(data_root) if f.endswith('.wav')] # len: 368

    for audio_name in tqdm(audio_list, desc="Audio Processing"): 
        audio_path = os.path.join(data_root, audio_name)
        wav_path = os.path.join(wav_root, audio_name)
        mel_path = os.path.join(mel_root, audio_name[:-4] + '.npy')

        # 1. resampling (16000 Hz)
        command = f"ffmpeg -loglevel error -y -i {audio_path} -ac 1 -acodec pcm_s16le -ar 16000 {wav_path}"
        output = subprocess.call(command, shell=True, stdout=None)

        # 2. extract mel-spectrogram
        wav = audio.load_wav(wav_path, 16000)
        mel = audio.melspectrogram(wav).T   # (T, 80)
        np.save(mel_path, mel)


if __name__ == '__main__':
	main(args)
