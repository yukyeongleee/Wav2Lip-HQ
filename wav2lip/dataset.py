import os
from glob import glob
import random
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_video_list(dataset_root, set_type):    # type: "train" or "val"
    videolist = []

    with open(f"assets/videolists/{set_type}.txt") as f:
        for line in f:
            line = line.strip()
            if ' ' in line: line = line.split()[0]
            videolist.append(os.path.join(dataset_root, line))

    return videolist


class AudioVisualDataset(Dataset):
    def __init__(self, dataset_root, video_step_size, mel_step_size, img_size, fps, set_type, isMaster):
        self.video_path_list = get_video_list(dataset_root["video"], set_type)
        self.mel_root = dataset_root["audio"]

        self.fps = fps  # 25
        self.img_size = img_size
        
        # SyncNet related parameters
        self.video_step_size = video_step_size  # 5
        self.mel_step_size = mel_step_size      # 16     

        # self.transforms = transforms.Compose([
        #     transforms.Resize((256,256)),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        if isMaster:
            print(f"Dataset of {self.__len__()} videos constructed for the training.")

    def get_window(self, vid_name, starting_frame):
        """
            Get **self.video_step_size** frames starting from the **starting_frame** 
        """
        starting_id = int(os.path.basename(starting_frame).split('.')[0])

        frames_in_window = []
        for frame_id in range(starting_id, starting_id + self.video_step_size):
            frame = os.path.join(vid_name, '{}.jpg'.format(frame_id))
            if not os.path.isfile(frame): 
                raise Exception("Missing Frame Exception") 
            frames_in_window.append(frame)

        return frames_in_window

    def read_window(self, frames_in_window):
        """
            Read images whose name is in the **frames_in_window** 
        """
        if frames_in_window is None: 
            raise Exception("Empty Window Exception")

        window = []
        for frame_name in frames_in_window:
            img = cv2.imread(frame_name)[:, :, ::-1] # BGR(cv2) to RGB
            if img is None:
                raise Exception("Missing Frame Exception")

            img = cv2.resize(img, (self.img_size, self.img_size))
            # Q. Other transforms?
            window.append(img)

        # Postprocessing (from the  **prepare_window** function )
        # 3 x T x H x W
        window = np.asarray(window) / 255.  # [5, 96, 96, 3]
        window = np.transpose(window, (3, 0, 1, 2)) # [3, 5, 96, 96]

        return window

    def crop_mel(self, mel, starting_frame):
        """
            Take **self.mel_step_size** frames of **mel** starting from **starting_frame**   
        """
        # num_frames = (T x hop_size * fps) / sample_rate
        
        video_starting_id = starting_frame if isinstance(starting_frame, int) else int(os.path.basename(starting_frame).split('.')[0])
        mel_starting_id = int(80. * (video_starting_id / self.fps))

        return mel[mel_starting_id : mel_starting_id + self.mel_step_size, :]

    def concatenate_mel_segments(self, mel, starting_frame):
        """
            Crop and concatenate **self.mel_step_size** mel segments of varying starting_frame
        """
        video_starting_id = int(os.path.basename(starting_frame).split('.')[0])

        mel_segments = []

        if video_starting_id < 1:
            raise Exception("Invalid Starting Frame Exception") # 왜 0은 안되는지 모르겠음

        for i in range(video_starting_id, video_starting_id + self.video_step_size):
            m = self.crop_mel(mel, i - 1)  
            if m.shape[0] != self.mel_step_size:
                raise Exception("Short Mel Exception")
            mel_segments.append(m.T)

        mel_segments = np.asarray(mel_segments)

        return mel_segments   

    def __len__(self):
        return len(self.video_path_list)

    def __getitem__(self, item): 
        while True:
            # Choose the video
            item = random.randint(0, len(self.video_path_list) - 1)
            vid_name = self.video_path_list[item]
            img_names = list(glob(os.path.join(vid_name, '*.jpg')))
            
            if len(img_names) <= 3 * self.video_step_size:
                continue

            # Choose two starting frames (one for the positive, the other for the negative sample)
            pos_img_name = random.choice(img_names)
            neg_img_name = random.choice(img_names)
            while pos_img_name == neg_img_name:
                neg_img_name = random.choice(img_names)
            
            try: 
                # Read both positive and negative windows of length **self.video_step_size**
                pos_frame_names = self.get_window(vid_name, pos_img_name)
                neg_frame_names = self.get_window(vid_name, neg_img_name)
                pos_window = self.read_window(pos_frame_names)
                neg_window = self.read_window(neg_frame_names)

                # Load corresponding mel-spectrograms
                mel_path = os.path.join(self.mel_root, os.path.basename(vid_name)+".npy")
                orig_mel = np.load(mel_path) # (frames, 80)
                concatenated_mels = self.concatenate_mel_segments(orig_mel.copy(), pos_img_name)    # (16, 80, 16)
            except Exception as e:
                continue

            # Get corresponding mel-spectrogram segments
            mel = self.crop_mel(orig_mel.copy(), pos_img_name)
            if (mel.shape[0] != self.mel_step_size):
                continue
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            concatenated_mels = torch.FloatTensor(concatenated_mels).unsqueeze(1) # (16, 1, 80, 16)

            GT = pos_window.copy()
            GT = torch.FloatTensor(GT)

            pos_window[:, :, pos_window.shape[2]//2:] = 0. # mask the lower half
            X = np.concatenate([pos_window, neg_window], axis=0) # pos_window for head pose, neg_window for identity
            X = torch.FloatTensor(X)

            return X, concatenated_mels, mel, GT
