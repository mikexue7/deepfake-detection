import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

# from moviepy.editor import AudioFileClip

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

def train(model, optimizer, loader_train, device, epochs=1):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    acc = check_accuracy(model, loader_train, device)
    print('Training accuracy before training: %.2f' % (100 * acc))
    for e in range(epochs):
        print(f"Begin training for epoch {e + 1}")
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)

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
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                acc = check_accuracy(model, loader_train, device)
                print('Training accuracy: %.2f' % (100 * acc))

def check_accuracy(model, loader, device):
    num_correct, num_samples = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            preds = torch.round(torch.sigmoid(scores))
            num_correct += (preds == y).sum()
            num_samples += preds.shape[0]
        acc = float(num_correct) / num_samples
    return acc