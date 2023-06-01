import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import cv2
import os
import json
import random
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/michaelxue/anaconda3/lib/python3.9/site-packages/ffmpeg/"
# from moviepy.editor import AudioFileClip

import pdb
import argparse

from frame_by_frame_model import flatten_videos_and_labels, unflatten_probs_and_labels, FrameByFrameCNN
from utils import train, eval_model, extract_faces_square, merge_metadata, calc_neg_to_pos_sample_ratio
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from early_fusion import EarlyFusion
from late_fusion import LateFusion

TRAIN_DATA_DIRECTORY = "./dfdc_train_part_"
ASPECT_RATIO = 16 / 9 # 1920 / 1080
VIDEOS_PROCESS_AT_ONCE = 400
TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15]
EMBEDDING_SIZE = 512

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

def load_frames(filepath, fr_downsample_factor, face_detector):
    # Open the video file
    cap = cv2.VideoCapture(filepath)

    frames = []
    counter = 0
    # Loop through the video frames
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        # keep every fr_downsample_factor frame
        if counter % fr_downsample_factor == 0:
            frames.append(frame)
        counter += 1

    # in future can extract frame by frame, and if certain frames aren't picked up by detector, can pad with 0s to make all videos same length
    # for now it seems that face detector picks up almost every video completely
    extracted_face_frames = [frame.unsqueeze(dim=0) for frame in face_detector(frames)] # leads to error if certain frame cannot detect a face

    # Release the video file
    cap.release()

    return torch.cat(extracted_face_frames, dim=0)

def load_data_and_labels(files, metadata, face_detector):
    data = []
    labels = []
    files_kept = []
    for i, filepath in enumerate(files):
        try:
            frames = load_frames(filepath, fr_downsample_factor=5, face_detector=face_detector)
            files_kept.append(os.path.basename(filepath))
            # we will need to pad these later before passing into models, unless each video is same length, which seems to be the case
            data.append(frames)
            print("{}/{} files loaded".format(i + 1, len(files)))
        except:
            print("Could not load " + filepath)
            continue
    # load metadata info
    for file in files_kept:
        label = metadata[file]["label"]
        labels.append(1 if label == 'FAKE' else 0)
    
    # return torch.cat(data, dim=0)
    return pad_sequence(data, batch_first=True), torch.tensor(labels, dtype=torch.float32).unsqueeze(dim=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_videos', type=int, default=100,
                        help='number of videos to use in total (training and test)')
    parser.add_argument('--pretrained_path', type=str,
                        help='path to pretrained model state dict')
    args = parser.parse_args()
    num_videos = args.num_videos
    num_train = int(num_videos * TRAIN_VAL_TEST_SPLIT[0])
    num_val = int(num_videos * TRAIN_VAL_TEST_SPLIT[1])
    num_test = int(num_videos * TRAIN_VAL_TEST_SPLIT[2])

    files = [os.path.join(TRAIN_DATA_DIRECTORY + f"{i}", file) for i in range(5) for file in os.listdir(TRAIN_DATA_DIRECTORY + f"{i}/") if file.endswith('.mp4')]
    random.seed(231)
    random.shuffle(files)
    files = files[:num_videos]
    files_train, files_val, files_test = files[:num_train], files[num_train:num_train + num_val], files[num_train + num_val:] # split training/val/testing by files first, since we cannot fit all the data in memory
    metadata_files = [os.path.join(TRAIN_DATA_DIRECTORY + f"{i}/", "metadata.json") for i in range(5)]
    metadata_all = merge_metadata(metadata_files)
    # Create a generator that yields groups of videos at a time
    file_train_groups = (files_train[i:i + VIDEOS_PROCESS_AT_ONCE] for i in range(0, len(files_train), VIDEOS_PROCESS_AT_ONCE))
    # eventually we'll need one for file_test_groups

    # Training
    # Use a loop and next() to get all the groups of videos, train on each set of videos
    num_iters = 0
    device = torch.device('cuda') # or mps or cpu
    mtcnn = MTCNN(image_size=160, post_process=True, select_largest=False, device=device)

    # load val dataset
    val_data, val_labels = load_data_and_labels(files_val, metadata_all, mtcnn)
    val_dataset = DeepfakeDataset(val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    best_val = 0
    best_model_state_dict = None
    pos_weight = torch.tensor([calc_neg_to_pos_sample_ratio(metadata_all)]).to(device)
    # load training data over time
    while True:
        try:
            curr_files = next(file_train_groups) # get next batch of files
            train_data, train_labels = load_data_and_labels(curr_files, metadata_all, mtcnn)
            print(f"Input has shape {train_data.shape}, labels have shape {train_labels.shape}")
            train_dataset = DeepfakeDataset(train_data, train_labels)
            train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

            num_frames = train_data.shape[1]

            # initialize model and optimizer, then train
            if num_iters == 0:
                resnet = InceptionResnetV1(pretrained='vggface2') # pretrained facial recognition model
                resnet.load_state_dict(torch.load(args.pretrained_path), strict=False) # only load resnet part of model
                # freeze resnet
                for param in resnet.parameters():
                    param.requires_grad = False
                # model = LateFusion(resnet, num_frames, EMBEDDING_SIZE, [256, 128]).to(device)
                model = EarlyFusion(resnet, num_frames, EMBEDDING_SIZE, [256, 128]).to(device)
                # model = nn.Sequential(resnet, nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
                total_params = sum(param.numel() for param in model.parameters())
                print(f"Number of model parameters: {total_params}")
                optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
                # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            best_val_i, best_model_state_dict_i = train(model, optimizer, pos_weight, train_dataloader, val_dataloader, device=device, epochs=5)
            # best_val_i, best_model_state_dict_i = train(model, optimizer, loss_fn, train_dataloader, val_dataloader, device=device, epochs=5, preprocess_fn=flatten_videos_and_labels, postprocess_fn=unflatten_probs_and_labels)
            if best_val_i > best_val:
                best_val = best_val_i
                best_model_state_dict = best_model_state_dict_i

            num_iters += 1

        except StopIteration:
            # Handle the end of the sequence
            break

    # Evaluation on test set
    test_data, test_labels = load_data_and_labels(files_test, metadata_all, mtcnn)
    test_dataset = DeepfakeDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    model.load_state_dict(best_model_state_dict) # load best model
    test_acc, test_loss, y_true, y_pred = eval_model(model, test_dataloader, device=device)
    # test_acc, test_loss, y_true, y_pred = eval_model(model, test_dataloader, device=device, preprocess_fn=flatten_videos_and_labels, postprocess_fn=unflatten_probs_and_labels)
    print("Test accuracy = %.4f, log loss = %.4f" % (100 * test_acc, test_loss))

    # cm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')
    # plt.show()