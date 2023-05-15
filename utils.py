import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2

# import resource

# from moviepy.editor import AudioFileClip

# def check_memory_usage():
#     usage = resource.getrusage(resource.RUSAGE_SELF)
#     memory_usage = usage[2] / (1024 ** 3)
#     print(f"Current process is using {memory_usage:.2f} GB of memory.")

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def train(model, optimizer, loader_train, device, epochs, preprocess_fn=None, postprocess_fn=None):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    model.train()  # put model to training mode
    for e in range(epochs):
        print(f"Begin training for epoch {e + 1}")
        for t, (x, y) in enumerate(loader_train):
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)

            if preprocess_fn:
                x, y = preprocess_fn(x, y)
            scores = model(x)
            loss = F.binary_cross_entropy(torch.sigmoid(scores), y, reduction='mean') # average BCE loss per frame (for FBF model)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            print('Iteration %d, loss = %.4f' % (t, loss.item())) # generic loss, can differ between models based on preprocessing
                
        acc, log_loss, _, _ = eval_model(model, loader_train, device, preprocess_fn, postprocess_fn)
        print('Training accuracy = %.4f, log loss = %.4f' % (100 * acc, log_loss)) # official log loss score

def eval_model(model, loader, device, preprocess_fn=None, postprocess_fn=None):
    num_correct, num_samples, total_loss = 0, 0, 0
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            batch_size = x.shape[0]
            x = x.to(device=device)
            y = y.to(device=device)
            if preprocess_fn:
                x, y = preprocess_fn(x, y)
            scores = model(x)
            probs = torch.sigmoid(scores)
            if postprocess_fn:
                probs, y = postprocess_fn(probs, y, batch_size)
            preds = torch.round(probs)
            num_correct += (preds == y).sum()
            num_samples += batch_size
            total_loss += F.binary_cross_entropy(probs, y, reduction='sum')

            y_true.extend(y.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())
        acc = float(num_correct) / num_samples
        log_loss = total_loss.item() / num_samples
    return acc, log_loss, y_true, y_pred

# def eval_model_with_eval_fn(model, loader, device, eval_fn, preprocess_fn=None, postprocess_fn=None):
#     num_correct, num_samples, total_loss = 0, 0, 0
#     model.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for x, y in loader:
#             batch_size = x.shape[0]
#             x = x.to(device=device)
#             y = y.to(device=device)
#             if preprocess_fn:
#                 x, y = preprocess_fn(x, y)
#             scores = model(x)
#             if postprocess_fn:
#                 scores, y = postprocess_fn(scores, y, batch_size)
#             preds, nc, log_loss_batch = eval_fn(scores, y)
#             num_correct += nc
#             num_samples += batch_size
#             total_loss += log_loss_batch

#             y_true.extend(y.cpu().numpy().flatten())
#             y_pred.extend(preds.cpu().numpy().flatten())
#         acc = float(num_correct) / num_samples
#         log_loss = total_loss.item() / num_samples
#     return acc, log_loss, y_true, y_pred

def extract_faces(image):
    # Load the pre-trained face cascade from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Extract and return the faces
    extracted_faces = []
    for (x, y, w, h) in faces:
        face = image_rgb[y:y+h, x:x+w]
        extracted_faces.append(face)

    return extracted_faces

def extract_faces_square(image, dimension):
    # Load the pre-trained face cascade from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Extract and return the square faces with original color
    extracted_faces = []
    for (x, y, w, h) in faces:
        # Calculate the maximum dimension (width or height) of the face
        face_size = max(w, h)

        # Calculate the center of the face
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Calculate the coordinates for the square bounding box
        face_left = face_center_x - face_size // 2
        face_top = face_center_y - face_size // 2
        face_right = face_left + face_size
        face_bottom = face_top + face_size

        # Extract the face region and resize it to a square shape
        face = image_rgb[face_top:face_bottom, face_left:face_right]
        face = cv2.resize(face, (dimension, dimension))

        extracted_faces.append(face)

    return extracted_faces

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
#     width = 128
#     height = 128
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

#     for frame in frames:
#         out.write(frame)