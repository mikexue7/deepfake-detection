#from facenet_pytorch import InceptionResnetV1
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np
import os
import cv2
#from utils import extract_faces_square, train, eval_model


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

# in this module, we want to further pretrain a resnet on our deepfake image dataset
PATH_TO_TRAIN_IMAGES = '/users/jacobmejia/Downloads/Dataset/Train/'
PATH_TO_TEST_IMAGES = "/users/jacobmejia/Downloads/Dataset/Test/"

class DeepfakeImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
    
def load_data_and_labels(directory, num_samples):
    data = []
    labels = []

    fake_dir = os.path.join(directory, "Fake/")
    real_dir = os.path.join(directory, "Real/")
    # load fake images
    for i, file in enumerate(os.listdir(fake_dir)):
        if i >= num_samples:
            break
        filepath = os.path.join(fake_dir, file)
        img = cv2.imread(filepath)
        extracted_faces = extract_faces_square(img, 224)
        if len(extracted_faces) == 0:
            continue
        extracted_face = extracted_faces[0]
        img_resized = np.transpose(extracted_face, (2, 0, 1)).astype('float32')
        data.append(torch.from_numpy(img_resized).unsqueeze(dim=0))
        labels.append(1)
        if (i + 1) % 1000 == 0:
            print(f"Loaded {i + 1} fake images")
    # load real images; note we don't need to shuffle here, dataloader will take care of it during training
    for i, file in enumerate(os.listdir(real_dir)):
        if i >= num_samples:
            break
        filepath = os.path.join(real_dir, file)
        img = cv2.imread(filepath)
        extracted_faces = extract_faces_square(img, 224)
        if len(extracted_faces) == 0:
            continue
        extracted_face = extracted_faces[0]
        img_resized = np.transpose(extracted_face, (2, 0, 1)).astype('float32')
        data.append(torch.from_numpy(img_resized).unsqueeze(dim=0))
        labels.append(0)
        if (i + 1) % 1000 == 0:
            print(f"Loaded {i + 1} real images")
    
    return torch.cat(data, dim=0), torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

def return_data():
    train_data, train_labels = load_data_and_labels(PATH_TO_TRAIN_IMAGES, 10)
    print(f"Input has shape {train_data.shape}, labels have shape {train_labels.shape}")
    train_dataset = DeepfakeImageDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_dataloader