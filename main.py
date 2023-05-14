import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

import numpy as np
import cv2
import os
import json
import random
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/michaelxue/anaconda3/lib/python3.9/site-packages/ffmpeg/"
# from moviepy.editor import AudioFileClip

import pdb
import resource
import argparse

from frame_by_frame_model import flatten_videos_and_labels, fbf_eval, FrameByFrameCNN
from utils import train, check_accuracy, check_memory_usage, extract_faces_square

TRAIN_DATA_DIRECTORY = "./train_sample_videos/"
ASPECT_RATIO = 16 / 9 # 1920 / 1080
VIDEOS_PROCESS_AT_ONCE = 10
TRAIN_TEST_SPLIT = 0.8

class DeepfakeDataset(Dataset):
    def __init__(self, videos, labels):
        self.videos = videos
        self.labels = labels

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video = self.videos[idx]
        label = self.labels[idx]
        return video, label

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
            #frame_resized = np.transpose(cv2.resize(frame, (new_width, new_height)), (2, 0, 1)).astype('float32') # make channel first dimension
            extracted_face = extract_faces_square(frame)[0]
            frame_resized = np.transpose(extracted_face, (2, 0, 1)).astype('float32')
            frames.append(torch.from_numpy(frame_resized).unsqueeze(dim=0))
        counter += 1

    # Release the video file
    cap.release()

    return torch.cat(frames, dim=0)

def load_data_and_labels(files, metadata_path):
    data = []
    labels = []
    files_kept = []
    for i, file in enumerate(files):
        filepath = os.path.join(TRAIN_DATA_DIRECTORY, file)
        frames = load_frames(filepath, img_downsample_factor=4, fr_downsample_factor=5)
        if frames is None:
            continue
        files_kept.append(file) # only those matching the aspect ratio 16:9 are kept for now
        # we will need to pad these later before passing into models, unless each video is same length, which seems to be the case
        data.append(frames.unsqueeze(dim=0))
        print("{}/{} files loaded".format(i + 1, len(files)))
    # load metadata info
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        for file in files_kept:
            label = metadata[file]["label"]
            labels.append(1 if label == 'FAKE' else 0)
    
    return torch.cat(data, dim=0), torch.tensor(labels, dtype=torch.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_videos', type=int, default=100,
                        help='number of videos to use in total (training and test)')
    args = parser.parse_args()
    num_videos = args.num_videos
    num_train = int(num_videos * TRAIN_TEST_SPLIT)

    files = [file for file in os.listdir(TRAIN_DATA_DIRECTORY) if file.endswith('.mp4')][:num_videos]
    random.shuffle(files)
    files_train, files_test = files[:num_train], files[num_train:] # split training/testing by files first, since we cannot fit all the data in memory
    metadata_path = os.path.join(TRAIN_DATA_DIRECTORY, "metadata.json")
    # Create a generator that yields groups of videos at a time
    file_train_groups = (files_train[i:i + VIDEOS_PROCESS_AT_ONCE] for i in range(0, len(files_train), VIDEOS_PROCESS_AT_ONCE))
    # eventually we'll need one for file_test_groups

    # Training
    # Use a loop and next() to get all the groups of videos, train on each set of videos
    num_iters = 0
    while True:
        try:
            curr_files = next(file_train_groups) # get next batch of files
            train_data, train_labels = load_data_and_labels(curr_files, metadata_path)
            check_memory_usage()
            print(f"Input has shape {train_data.shape}, labels have shape {train_labels.shape}")
            train_dataset = DeepfakeDataset(train_data, train_labels)
            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

            height, width = train_data.shape[3], train_data.shape[4]

            device = torch.device('cpu') # or mps or cpu

            # initialize model and optimizer, then train
            if num_iters == 0:
                fbf_model = FrameByFrameCNN([32, 16], [5, 3], [2, 1], [100], 2, height, width)
                total_params = sum(param.numel() for param in fbf_model.parameters())
                print(f"Number of model parameters: {total_params}")
                check_memory_usage()
                optimizer = optim.Adam(fbf_model.parameters(), lr=1e-5)

            train(fbf_model, optimizer, train_dataloader, device=device, epochs=4, eval_fn=fbf_eval, preprocess_fn=flatten_videos_and_labels)

            num_iters += 1

        except StopIteration:
            # Handle the end of the sequence
            break

    # Evaluation on test set
    test_data, test_labels = load_data_and_labels(files_test, metadata_path)
    test_dataset = DeepfakeDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    test_acc, test_loss = check_accuracy(fbf_model, test_dataloader, device=device, eval_fn=fbf_eval, preprocess_fn=flatten_videos_and_labels)
    print("Test accuracy = %.4f, test log loss = %.4f" % (test_acc, test_loss))
