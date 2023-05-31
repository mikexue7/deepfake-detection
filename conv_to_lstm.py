import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvToLSTM(nn.Module):
    def __init__(self, pretrained_model, num_frames, embedding_size, hidden_sizes):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.num_frames = num_frames
        self.lstm = nn.LSTM(embedding_size, embedding_size, batch_first=True, dropout=0.5)

        self.fc_layers = nn.ModuleList()
        prev_hidden_size = embedding_size
        for hidden_size in hidden_sizes:
            self.fc_layers.append(nn.Linear(prev_hidden_size, hidden_size))
            prev_hidden_size = hidden_size
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, x): # x is (N, T, 3, H, W)
        frame_features = []
        for i in range(self.num_frames):
            frame = x[:, i, :, :, :]
            frame_features.append(self.pretrained_model(frame))
        
        frame_features = torch.cat(frame_features, dim=1) # (N, T, E)
        _, (h_n, _) = self.lstm(frame_features)

        x = h_n
        for fc_layer in self.fc_layers[:-1]:
            x = F.relu(fc_layer(x))
        score = self.fc_layers[-1](x)
        return score