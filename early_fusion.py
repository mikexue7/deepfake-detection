import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyFusion(nn.Module):
    def __init__(self, pretrained_model, num_frames, embedding_size, hidden_sizes):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.conv = nn.Conv2d(3 * num_frames, 3, 1, 1)
        self.fc_layers = nn.ModuleList()
        prev_hidden_size = embedding_size
        for hidden_size in hidden_sizes:
            self.fc_layers.append(nn.Linear(prev_hidden_size, hidden_size))
            prev_hidden_size = hidden_size
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, x): # x is (N, T, 3, H, W)
        x = x.reshape(x.shape[0], -1, *x.shape[3:])
        x = self.conv(x) # (N, 3, H, W)
        x = self.pretrained_model(x)
        for fc_layer in self.fc_layers[:-1]:
            x = F.relu(fc_layer(x))
        score = self.fc_layers[-1](x)
        return score