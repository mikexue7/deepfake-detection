import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np

# frames is (n_videos, n_frames, height, width, channels); we want each frame to correspond to its label
def preprocess_frames_and_labels(frames, labels):
    _, n_frames, H, W, C = frames.shape
    frames_reshaped = frames.reshape((-1, H, W, C))
    labels_reshaped = torch.repeat_interleave(labels, n_frames).unsqueeze(dim=1)
    assert frames_reshaped.shape[0] == labels_reshaped.shape[0]
    return frames_reshaped, labels_reshaped

# write dataset
class FrameByFrameDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        return frame, label

# model
def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    
class FrameByFrameCNN(nn.Module):
    def __init__(self, channel_sizes, kernel_sizes, paddings, hidden_sizes, input_height, input_width):
        super().__init__()
        assert len(channel_sizes) == len(kernel_sizes)
        assert len(kernel_sizes) == len(paddings)
        self.conv_layers = nn.ModuleList()
        prev_channel_size = 3
        for i, channel_size in enumerate(channel_sizes):
            self.conv_layers.append(nn.Conv2d(prev_channel_size, channel_size, kernel_size=kernel_sizes[i], padding=paddings[i]))
            prev_channel_size = channel_size
        self.fc_layers = nn.ModuleList()
        prev_hidden_size = channel_sizes[-1] * input_height * input_width // 4 # because of max pool, make this more elegant
        for i, hidden_size in enumerate(hidden_sizes):
            self.fc_layers.append(nn.Linear(prev_hidden_size, hidden_size))
            prev_hidden_size = hidden_size
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = flatten(x)
        for fc_layer in self.fc_layers[:-1]:
            x = F.relu(fc_layer(x))
        score = self.fc_layers[-1](x)
        return score
