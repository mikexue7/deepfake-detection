import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2

import resource

# from moviepy.editor import AudioFileClip

def check_memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    memory_usage = usage[2] / (1024 ** 3)
    print(f"Current process is using {memory_usage:.2f} GB of memory.")

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def train(model, optimizer, loader_train, device, epochs, eval_fn, preprocess_fn=None):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print(f"Begin training for epoch {e + 1}")
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)

            if preprocess_fn:
                x, y = preprocess_fn(x, y)
            scores = model(x)
            loss = F.binary_cross_entropy(torch.sigmoid(scores), y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % 10 == 0:
                print('Iteration %d, log loss = %.4f' % (t, loss.item()))
                
        acc, _ = check_accuracy(model, loader_train, device, eval_fn, preprocess_fn)
        print('Training accuracy = %.4f' % (100 * acc))

def check_accuracy(model, loader, device, eval_fn, preprocess_fn=None):
    num_correct, num_samples, total_loss = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            if preprocess_fn:
                x, y = preprocess_fn(x, y)
            scores = model(x)
            preds = torch.round(torch.sigmoid(scores))
            total_loss += F.binary_cross_entropy(torch.sigmoid(scores), y, reduction='sum')
            num_correct += eval_fn(preds, y)
            num_samples += preds.shape[0]
        acc = float(num_correct) / num_samples
        log_loss = total_loss.item() / num_samples
    return acc, log_loss

# def load_audio(filepath):
#     # Load the audio file
#     audio_clip = AudioFileClip(filepath)

#     # Extract the audio data
#     audio_data = audio_clip.to_soundarray()

#     # Print the shape of the audio data array
#     print(audio_data.shape)

#     # Release the audio file
#     audio_clip.close()

# testing video output
# def write_to_video(frames):
#     fps = 10
#     width = 480
#     height = 270
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

#     for frame in frames:
#         out.write(frame)