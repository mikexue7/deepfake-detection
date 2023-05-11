import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import cv2
import os
import json
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/michaelxue/anaconda3/lib/python3.9/site-packages/ffmpeg/"
# from moviepy.editor import AudioFileClip

import pdb
import resource

from frame_by_frame_model import preprocess_frames_and_labels, FrameByFrameDataset, FrameByFrameCNN
from utils import train

TRAIN_DATA_DIRECTORY = "./train_sample_videos/"
ASPECT_RATIO = 16 / 9 # 1920 / 1080

def load_frames(filepath, img_downsample_factor, fr_downsample_factor):
    # Open the video file
    cap = cv2.VideoCapture(filepath)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    aspect_ratio = width / height
    # for now, we only keep videos with aspect ratio of 16:9; this is the majority of videos
    if aspect_ratio != ASPECT_RATIO:
        return None

    frames = []
    counter = 0
    # Loop through the video frames
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        # keep every fr_downsample_factor frame
        if counter % fr_downsample_factor == 0:
            new_height = frame.shape[0] // img_downsample_factor
            new_width = int(new_height * ASPECT_RATIO)
            frame_resized = np.transpose(cv2.resize(frame, (new_width, new_height)), (2, 0, 1)).astype('float32') # make channel first dimension
            frames.append(torch.from_numpy(frame_resized).unsqueeze(dim=0))
        counter += 1

    # Release the video file
    cap.release()

    return torch.cat(frames, dim=0)

# testing video output
# def write_to_video(frames):
#     fps = 10
#     width = 480
#     height = 270
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

#     for frame in frames:
#         out.write(frame)

def load_data_and_labels(directory):
    data = []
    labels = []
    files_kept = []
    num_files = len(os.listdir(directory))
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith('.mp4'): 
            filepath = os.path.join(directory, filename)
            frames = load_frames(filepath, img_downsample_factor=4, fr_downsample_factor=5)
            if frames is None:
                continue
            # write_to_video(frames)
            files_kept.append(filename)
            # we will need to pad these later before passing into models, unless each video is same length, which seems to be the case
            data.append(frames.unsqueeze(dim=0))
            print("{}/{} files loaded".format(i + 1, num_files - 1))
        if len(data) == 5: # for now, let's just use 5 training files
            break
    # load metadata info
    metadata_path = os.path.join(directory, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        for filename in files_kept:
            label = metadata[filename]["label"]
            labels.append(1 if label == 'FAKE' else 0)
    
    return torch.cat(data, dim=0), torch.tensor(labels, dtype=torch.float32)

def check_memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    memory_usage = usage[2] / (1024 ** 3)
    print(f"Current process is using {memory_usage:.2f} GB of memory.")

# def load_audio(filepath):
#     # Load the audio file
#     audio_clip = AudioFileClip(filepath)

#     # Extract the audio data
#     audio_data = audio_clip.to_soundarray()

#     # Print the shape of the audio data array
#     print(audio_data.shape)

#     # Release the audio file
#     audio_clip.close()

# WE MAY NEED TO DOWNSAMPLE VIDEOS, RIGHT NOW THEY ARE HD AND TAKE UP A LOT OF SPACE
# AND/OR WE CAN TAKE A SUBSET OF FRAMES PER VIDEO (take every other frame)
# CROP THE FACE (this gives more flexibility for sizing frames)? 
# but there could be multiple faces, and perhaps other signs from the video that aren't the face.
# we could investigate what parts of image are being detected from the model based on features
if __name__ == '__main__':
    train_data, train_labels = load_data_and_labels(TRAIN_DATA_DIRECTORY)
    check_memory_usage()
    print(train_data.shape, train_labels.shape)
    train_frames_fbf, train_labels_fbf = preprocess_frames_and_labels(train_data, train_labels)
    print(train_frames_fbf.shape, train_labels_fbf.shape)
    fbf_dataset = FrameByFrameDataset(train_frames_fbf, train_labels_fbf)
    train_dataloader = DataLoader(fbf_dataset, batch_size=64, shuffle=True)

    # initialize model and optimizer, then train
    fbf_model = FrameByFrameCNN([32, 16], [5, 3], [2, 1], [100], train_frames_fbf.shape[2], train_frames_fbf.shape[3])
    total_params = sum(param.numel() for param in fbf_model.parameters())
    print(f"Number of model parameters: {total_params}")
    check_memory_usage()
    optimizer = optim.Adam(fbf_model.parameters())

    train(fbf_model, optimizer, train_dataloader, 'cpu')

