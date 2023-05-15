import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
from utils import flatten

# frames is (n_videos, n_frames, channels, height, width); we want each frame to correspond to its label
def flatten_videos_and_labels(videos, labels):
    _, n_frames, C, H, W = videos.shape
    videos_reshaped = videos.reshape((-1, C, H, W))
    labels_reshaped = torch.repeat_interleave(labels, n_frames).unsqueeze(dim=1)
    assert videos_reshaped.shape[0] == labels_reshaped.shape[0]
    return videos_reshaped, labels_reshaped

def unflatten_probs_and_labels(probs, labels, batch_size):
    # probs, labels are (n_videos * n_frames), need to reshape them to (n_videos)
    probs_reshaped = torch.mean(probs.reshape((batch_size, -1)), dim=1) # (n_videos), take the average of the probabilities
    labels_reshaped = labels.reshape((batch_size, -1))[:, 0] # each column is the same, has shape (n_videos)
    return probs_reshaped, labels_reshaped

# FOR MAJORITY VOTE
# def fbf_eval(scores, labels): # scores here is (n_videos, n_frames), labels is (n_videos)
#     frame_preds = torch.round(torch.sigmoid(scores))
#     video_preds = torch.round(torch.sum(frame_preds, dim=1) / scores.shape[1]) # majority vote

#     num_correct = (video_preds == labels).sum()
#     scores_per_video = torch.mean(scores, dim=1)
#     log_loss_batch = F.binary_cross_entropy(torch.sigmoid(scores_per_video), labels, reduction='sum')
#     return video_preds, num_correct, log_loss_batch

# model
class FrameByFrameCNN(nn.Module):
    def __init__(self, channel_sizes, kernel_sizes, paddings, hidden_sizes, max_pool_downsample_factor, input_height, input_width):
        super().__init__()
        assert len(channel_sizes) == len(kernel_sizes)
        assert len(kernel_sizes) == len(paddings)
        self.conv_layers = nn.ModuleList()
        prev_channel_size = 3
        for i, channel_size in enumerate(channel_sizes):
            self.conv_layers.append(nn.Conv2d(prev_channel_size, channel_size, kernel_size=kernel_sizes[i], padding=paddings[i]))
            prev_channel_size = channel_size
        self.fc_layers = nn.ModuleList()
        prev_hidden_size = channel_sizes[-1] * input_height * input_width // (max_pool_downsample_factor ** 2)
        for i, hidden_size in enumerate(hidden_sizes):
            self.fc_layers.append(nn.Linear(prev_hidden_size, hidden_size))
            prev_hidden_size = hidden_size
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], 1))

        self.max_pool_downsample_factor = max_pool_downsample_factor

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
        x = F.max_pool2d(x, kernel_size=self.max_pool_downsample_factor, stride=self.max_pool_downsample_factor)
        x = flatten(x)
        for fc_layer in self.fc_layers[:-1]:
            x = F.relu(fc_layer(x))
        score = self.fc_layers[-1](x)
        return score
