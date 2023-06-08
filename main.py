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
import matplotlib.pyplot as plt
import multiprocessing

import pdb
import argparse
import logging
import time

from frame_by_frame_model import flatten_videos_and_labels, unflatten_probs_and_labels, FrameByFrameCNN
from utils import train, eval_model, extract_faces_square, merge_metadata, balance_dataset, calc_neg_to_pos_sample_ratio, plot_visualizations
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from early_fusion import EarlyFusion
from late_fusion import LateFusion
from conv_to_lstm import ConvToLSTM

TRAIN_DATA_DIRECTORY = "./dfdc_train_part_"
ASPECT_RATIO = 16 / 9 # 1920 / 1080
VIDEOS_PROCESS_AT_ONCE = 500
TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15]
EMBEDDING_SIZE = 512
NUM_FOLDERS = 50
MAX_NUM_FRAMES = 60

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
    extracted_face_frames = [face_detector(frame).unsqueeze(dim=0) for frame in frames] # leads to error if certain frame cannot detect a face

    # Release the video file
    cap.release()

    return torch.cat(extracted_face_frames, dim=0)

def load_data_and_labels(files, metadata, face_detector):
    data = []
    labels = []
    files_kept = []

    neg_examples, pos_examples = 0, 0
    for i, filepath in enumerate(files):
        try:
            frames = load_frames(filepath, fr_downsample_factor=5, face_detector=face_detector)
            files_kept.append(os.path.basename(filepath))
            # we will need to pad these later before passing into models, unless each video is same length, which seems to be the case
            frames = frames[:MAX_NUM_FRAMES]
            data.append(frames)
            print("{}/{} files loaded".format(i + 1, len(files)))
        except:
            print("Could not load " + filepath)
            continue
    # load metadata info
    for file in files_kept:
        label = metadata[file]["label"]
        labels.append(1 if label == 'FAKE' else 0)
    
    return pad_sequence(data, batch_first=True), torch.tensor(labels, dtype=torch.float32).unsqueeze(dim=1)

def load_data_and_labels_multiprocessing(files, metadata, face_detector):
    # Number of processes to use
    num_processes = multiprocessing.cpu_count()

    # Calculate the number of files per process
    files_per_process = len(files) // num_processes

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Use the pool to asynchronously load data from files
    file_chunks = [files[i:i + files_per_process] for i in range(0, len(files), files_per_process)]
    results = []
    for chunk in file_chunks:
        results.append(pool.apply_async(load_data_and_labels, args=(chunk, metadata, face_detector)))
    
    # Get the loaded data from all processes
    loaded_data = []
    loaded_labels = []
    for result in results:
        data, labels = result.get()
        loaded_data.extend(data)
        loaded_labels.extend(labels)

    # Close the pool of worker processes
    pool.close()
    pool.join()

    return torch.cat(loaded_data, dim=0), torch.cat(loaded_labels, dim=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_videos', type=int, default=100,
                        help='number of videos to use in total (training and test)')
    parser.add_argument('--model', type=str, default='fbf',
                        help='type of model to use (fbf, early_fusion, late_fusion, lstm, transformer)')
    parser.add_argument('--pretrained_path', type=str,
                        help='path to pretrained model state dict')
    parser.add_argument('--log_output', type=str,
                        help='file for output logs')
    args = parser.parse_args()
    logging.basicConfig(filename=args.log_output, format='%(message)s', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    num_videos = args.num_videos
    num_train = int(num_videos * TRAIN_VAL_TEST_SPLIT[0])
    num_val = int(num_videos * TRAIN_VAL_TEST_SPLIT[1])
    num_test = int(num_videos * TRAIN_VAL_TEST_SPLIT[2])

    metadata_files = [os.path.join(TRAIN_DATA_DIRECTORY + f"{i}/", "metadata.json") for i in range(NUM_FOLDERS)]
    metadata_all = merge_metadata(metadata_files)

    files = [os.path.join(TRAIN_DATA_DIRECTORY + f"{i}", file) for i in range(NUM_FOLDERS) for file in sorted(os.listdir(TRAIN_DATA_DIRECTORY + f"{i}/")) if file.endswith('.mp4')] # use sorted to make files the same order across runs
    files = balance_dataset(files, metadata_all)
    random.seed(231)
    random.shuffle(files)
    files = files[:num_videos]
    files_train, files_val, files_test = files[:num_train], files[num_train:num_train + num_val], files[num_train + num_val:] # split training/val/testing by files first, since we cannot fit all the data in memory
    # Create a generator that yields groups of videos at a time
    file_train_groups = (files_train[i:i + VIDEOS_PROCESS_AT_ONCE] for i in range(0, len(files_train), VIDEOS_PROCESS_AT_ONCE))
    # eventually we may need one for file_test_groups

    # Training
    # Use a loop and next() to get all the groups of videos, train on each set of videos
    model_name = args.model
    
    num_iters = 0
    device = torch.device('cuda') # or mps or cpu
    mtcnn = MTCNN(image_size=160, post_process=True, select_largest=False, device=device) # need to use cpu here for multiprocessing

    # load val dataset
    val_data, val_labels = load_data_and_labels(files_val, metadata_all, mtcnn)
    val_dataset = DeepfakeDataset(val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    best_val = 0
    best_model_state_dict = None
    pos_weight = torch.tensor([calc_neg_to_pos_sample_ratio(metadata_all)]).to(device)
    print(f"Ratio of negative to positive samples: {pos_weight.item():.2f}")

    start_time = time.time()
    # load training data over time
    while True:
        try:
            curr_files = next(file_train_groups) # get next batch of files
            train_data, train_labels = load_data_and_labels(curr_files, metadata_all, mtcnn)
            print(f"Input has shape {train_data.shape}, labels have shape {train_labels.shape}")
            train_dataset = DeepfakeDataset(train_data, train_labels)
            train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

            num_frames = train_data.shape[1]

            # initialize model and optimizer, then train
            if num_iters == 0:
                resnet = InceptionResnetV1(pretrained='vggface2') # pretrained facial recognition model
                if args.pretrained_path:
                    resnet.load_state_dict(torch.load(args.pretrained_path), strict=False) # only load resnet part of model
                    # freeze resnet
                    for param in resnet.parameters():
                        param.requires_grad = False

                if model_name == 'fbf':
                    model = nn.Sequential(resnet, nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
                elif model_name == 'early_fusion':
                    model = EarlyFusion(resnet, num_frames, EMBEDDING_SIZE, [256, 128]).to(device)
                elif model_name == 'late_fusion':
                    model = LateFusion(resnet, num_frames, EMBEDDING_SIZE, [256, 128]).to(device)
                elif model_name == 'lstm':
                    model = ConvToLSTM(resnet, num_frames, EMBEDDING_SIZE, [256, 128]).to(device)
    
                total_params = sum(param.numel() for param in model.parameters())
                print(f"Number of model parameters: {total_params}")
                optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.001)
                # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            if model_name == 'fbf':
                best_val_i, best_model_state_dict_i = train(model, optimizer, pos_weight, train_dataloader, val_dataloader, device=device, epochs=10, preprocess_fn=flatten_videos_and_labels, postprocess_fn=unflatten_probs_and_labels)
            else:
                best_val_i, best_model_state_dict_i = train(model, optimizer, pos_weight, train_dataloader, val_dataloader, device=device, epochs=10)
            
            if best_val_i >= best_val: # for now, can change back to >
                best_val = best_val_i
                best_model_state_dict = best_model_state_dict_i

            num_iters += 1
            del train_data, train_labels, train_dataset, train_dataloader # free up space

        except StopIteration:
            # Handle the end of the sequence
            break

    end_time = time.time()
    elapsed_time_hours = (end_time - start_time) / 3600

    logger.info(f"Training took {elapsed_time_hours:.2f} hours")

    del val_data, val_labels, val_dataset, val_dataloader # free up space

    # Evaluation on test set
    test_data, test_labels = load_data_and_labels(files_test, metadata_all, mtcnn)
    test_dataset = DeepfakeDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    model.load_state_dict(best_model_state_dict) # load best model
    if model_name == 'fbf':
        test_acc, test_loss, y_true, y_pred, y_scores = eval_model(model, test_dataloader, device=device, preprocess_fn=flatten_videos_and_labels, postprocess_fn=unflatten_probs_and_labels)
    else:
        test_acc, test_loss, y_true, y_pred, y_scores = eval_model(model, test_dataloader, device=device)
    logger.info("Test accuracy = %.4f, log loss = %.4f" % (100 * test_acc, test_loss))

    plot_visualizations(y_true, y_pred, y_scores, model_name)