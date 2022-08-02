import os
import argparse
import subprocess
import cv2

from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument('--data_root', 	type=str, default="/data1/HDTF/", help='');
# parser.add_argument('--convert_fps', action='store_true', help='');
parser.add_argument('--frame_rate', type=int, default=25, help='Frame rate');
args = parser.parse_args();

setattr(args,'fps_root', f"/home/leee/data/HDTF/video-{args.frame_rate}fps/")

def get_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps

def convert_frame_rate(fps, src_path, dst_path):
    command = f"ffmpeg -loglevel error -i {src_path} -r {fps} {dst_path} -y"
    output = subprocess.call(command, shell=True, stdout=None)
    
def main(args):
    data_root = args.data_root # "/data1/HDTF/"
    # if args.convert_fps:
    fps_root = args.fps_root # f"/home/leee/data/HDTF/video-{args.frame_rate}fps/"
    os.makedirs(fps_root, exist_ok=True)

    video_list = [f for f in os.listdir(data_root) if f.endswith('.mp4')] 

    for video_name in tqdm(video_list, desc="Video Processing"): 
        src_path = os.path.join(data_root, video_name)
        dst_path = os.path.join(fps_root, video_name)

        # if args.convert_fps:
        if get_frame_rate(src_path) == args.frame_rate:
            command = f"cp {src_path} {dst_path}"
            output = subprocess.call(command, shell=True, stdout=None)
            continue          
        else:
            convert_frame_rate(args.frame_rate, src_path, dst_path)


if __name__ == '__main__':
	main(args)
