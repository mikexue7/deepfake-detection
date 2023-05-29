import torch
import torch.nn as nn
import torch.nn.functional as F

class LateFusion(nn.Module):
    def __init__(self, pretrained_model, num_frames, img_width):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.num_frames = num_frames
        self.conv = nn.Conv2d(3 * num_frames, 3, img_width, img_width)

    def forward(self, x): # x is (N, T, 3, H, W)
        x = x.flatten(-1, x.shape[2:])
        x = self.conv(x) # (N, 3, H, W)
        score = self.pretrained_model(x)
        return score