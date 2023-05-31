import json
from PIL import Image
import torch
from torchvision import transforms
import certifi
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import os
import cv2
from utils import train, eval_model
import pdb
from pytorch_pretrained_vit import ViT

import argparse

# in this module, we want to further pretrain a resnet on our deepfake image dataset
PATH_TO_TRAIN_IMAGES = "Train/"
PATH_TO_VAL_IMAGES = "Validation/"
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
    
def load_data_and_labels(directory, num_samples, device):
    data = []
    labels = []

    fake_dir = os.path.join(directory, "Fake/")
    real_dir = os.path.join(directory, "Real/")
    print("Gathering Data")
    # 160 is default value for passing into resnet after, post_process normalizes input, select_largest=False chooses face with highest prob rather than largest
    mtcnn = MTCNN(image_size=160, post_process=True, select_largest=False, device=device)

    # load fake images
    for i, file in enumerate(os.listdir(fake_dir)):
        if i >= num_samples:
            break
        filepath = os.path.join(fake_dir, file)
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # change to RGB
        img = Image.fromarray(img)
        extracted_face = mtcnn(img) # img here must have channels as last dim; note there are bugs that come up when processing in batches
        if extracted_face is None: # no face detected
            continue
        data.append(extracted_face.unsqueeze(dim=0)) # extracted face already a FloatTensor
        labels.append(1)
        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1} fake images")
    # load real images; note we don't need to shuffle here, dataloader will take care of it during training
    for i, file in enumerate(os.listdir(real_dir)):
        if i >= num_samples:
            break
        filepath = os.path.join(real_dir, file)
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # change to RGB
        img = Image.fromarray(img)
        extracted_face = mtcnn(img) # img here must have channels as last dim
        if extracted_face is None: # no face detected
            continue
        data.append(extracted_face.unsqueeze(dim=0)) # extracted face already a FloatTensor
        labels.append(0)
        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1} real images")
    
    return torch.cat(data, dim=0), torch.tensor(labels, dtype=torch.float32).unsqueeze(dim=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True, help='path to save pretrained CNN')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, train_labels = load_data_and_labels(PATH_TO_TRAIN_IMAGES, 500, device) # change num training images from fake + real
    print(f"Input has shape {train_data.shape}, labels have shape {train_labels.shape}") 
    train_dataset = DeepfakeImageDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    val_data, val_labels = load_data_and_labels(PATH_TO_VAL_IMAGES, 100, device) # change num val images
    val_dataset = DeepfakeImageDataset(val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)

    ViT = ViT('B_16_imagenet1k', pretrained=True, image_size=160) # pretrained ImageNet Model 
    model = nn.Sequential(
    ViT,
    nn.Linear(1000,128),
    nn.ReLU(),
    nn.Linear(128,1)
    ).to(device) 
    total_params = sum(param.numel() for param in ViT.parameters())
    print(f"Number of model parameters: {total_params}")
    optimizer = optim.Adam(ViT.parameters(), lr=1e-5, weight_decay=0.001)

    loss_fn = nn.BCELoss()
    _, best_model_state_dict = train(model, optimizer,loss_fn ,train_dataloader, val_dataloader, device, epochs=10)
    model.load_state_dict(best_model_state_dict)

    # evaluate on test set
    test_data, test_labels = load_data_and_labels(PATH_TO_TEST_IMAGES, 5000, device)
    test_dataset = DeepfakeImageDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    test_acc, test_loss, y_true, y_pred = eval_model(model, test_dataloader, device=device)
    print("Test accuracy = %.4f, log loss = %.4f" % (100 * test_acc, test_loss))

    torch.save(best_model_state_dict, args.save_path)
