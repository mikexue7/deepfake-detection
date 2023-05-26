from facenet_pytorch import InceptionResnetV1
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np
import os
import cv2
from utils import extract_faces_square, train, eval_model

# in this module, we want to further pretrain a resnet on our deepfake image dataset
PATH_TO_TRAIN_IMAGES = "pretrain_images/train/"
PATH_TO_TEST_IMAGES = "pretrain_images/test/"

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

    fake_dir = os.path.join(directory, "fake/")
    real_dir = os.path.join(directory, "real/")
    # load fake images
    for i, file in enumerate(os.listdir(fake_dir)):
        if i >= num_samples:
            break
        filepath = os.path.join(fake_dir, file)
        img = cv2.imread(filepath)
        extracted_faces = extract_faces_square(img, 160)
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
        extracted_faces = extract_faces_square(img, 160)
        if len(extracted_faces) == 0:
            continue
        extracted_face = extracted_faces[0]
        img_resized = np.transpose(extracted_face, (2, 0, 1)).astype('float32')
        data.append(torch.from_numpy(img_resized).unsqueeze(dim=0))
        labels.append(0)
        if (i + 1) % 1000 == 0:
            print(f"Loaded {i + 1} real images")
    
    return torch.cat(data, dim=0), torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

if __name__ == '__main__':
    train_data, train_labels = load_data_and_labels(PATH_TO_TRAIN_IMAGES, 5000)
    print(f"Input has shape {train_data.shape}, labels have shape {train_labels.shape}")
    train_dataset = DeepfakeImageDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    device = torch.device('mps')

    resnet = InceptionResnetV1(pretrained='vggface2') # pretrained facial recognition model
    model = nn.Sequential(resnet, nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))
    total_params = sum(param.numel() for param in resnet.parameters())
    print(f"Number of model parameters: {total_params}")
    optimizer = optim.Adam(resnet.parameters(), lr=1e-5)

    train(model, optimizer, train_dataloader, device, epochs=5)

    # evaluate on test set
    test_data, test_labels = load_data_and_labels(PATH_TO_TEST_IMAGES, 1000)
    test_dataset = DeepfakeImageDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    test_acc, test_loss, y_true, y_pred = eval_model(model, test_dataloader, device=device)
    print("Test accuracy = %.4f, log loss = %.4f" % (test_acc, test_loss))
