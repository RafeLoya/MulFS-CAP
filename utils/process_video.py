import argparse
import cv2
import sys
import os

def extract_frames(video_path, output_folder, prefix="frame"):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filename = os.path.join(output_folder, f"{prefix}{index:04d}.png")
        cv2.imwrite(filename, gray_frame)
        index += 1
    cap.release()
    print(f"Extracted {index} frames from {video_path} to {output_folder}")

# Example usage:
# extract_frames("./data/VIS1.MP4", "./data/test_VIS", "vis")
# extract_frames("./data/IR1.avi", "./data/test_IR", "ir")

parser = argparse.ArgumentParser(
    prog='Video Process',
    description='Dumps frames from given video to target directory'
)
parser.add_argument('video_path', help='path to video')
parser.add_argument('-o', '--output_folder', default='./data/test', help='folder to store dumped frames')
parser.add_argument('-f', '--frame_prefix', default='frame', help='prefix for the names of individual frames')

args = parser.parse_args()
extract_frames(args.video_path, args.output_folder, args.frame_prefix)
