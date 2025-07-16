import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed()

class GenericImageDataset(Dataset):
    def __init__(self, file_paths, labels, mode="train"):
        self.file_paths = file_paths
        self.labels = labels
        self.mode = mode

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert('RGB')
        transform = self.train_transform if self.mode == "train" else self.val_transform
        img = transform(img)
        return img, self.labels[idx]


class MedMNISTDataset(Dataset):

    def __init__(self, images, labels, mode="train"):
        self.images = images
        self.labels = labels
        self.mode = mode

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.mode == "train":
            img = self.train_transform(img)
        else:
            img = self.val_transform(img)
        return img, self.labels[idx]

def load_fetal_planes_db(root_dir):

    csv_path = os.path.join(root_dir, 'FETAL_PLANES_DB_data.csv')
    images_dir = os.path.join(root_dir, 'Images')

    df = pd.read_csv(csv_path, sep=';')

    image_paths = {}
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(filename)[0]
            image_paths[base_name] = os.path.join(images_dir, filename)

    file_paths, labels, train_flags = [], [], []
    for _, row in df.iterrows():
        base_name = os.path.splitext(row['Image_name'])[0]
        if base_name in image_paths:
            file_paths.append(image_paths[base_name])
            labels.append(row['Plane'])
            train_flags.append(row['Train '])

    label_map = {l: idx for idx, l in enumerate(sorted(set(labels)))}
    labels = [label_map[l] for l in labels]

    train_files = [p for p, f in zip(file_paths, train_flags) if f == 1]
    train_labels = [l for l, f in zip(labels, train_flags) if f == 1]
    test_files = [p for p, f in zip(file_paths, train_flags) if f == 0]
    test_labels = [l for l, f in zip(labels, train_flags) if f == 0]

    return train_files, train_labels, test_files, test_labels, label_map

def load_kvasir_v2(root_dir):

    train_files, test_files = [], []
    train_labels, test_labels = [], []
    seed = 42

    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        img_files = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not img_files:
            continue

        class_train, class_test = train_test_split(img_files, test_size=0.2, random_state=seed)

        train_files.extend(class_train)
        train_labels.extend([class_name] * len(class_train))

        test_files.extend(class_test)
        test_labels.extend([class_name] * len(class_test))

    label_map = {l: idx for idx, l in enumerate(sorted(set(train_labels + test_labels)))}
    train_labels = [label_map[l] for l in train_labels]
    test_labels = [label_map[l] for l in test_labels]

    return train_files, train_labels, test_files, test_labels, label_map

def load_medmnist_npz(npz_path):
    data = np.load(npz_path)

    def to_tensor(images, labels):
        images = torch.from_numpy(images).float().unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
        labels = torch.from_numpy(labels).long().squeeze()
        return images, labels

    train_images, train_labels = to_tensor(data['train_images'], data['train_labels'])
    val_images, val_labels = to_tensor(data['val_images'], data['val_labels'])
    test_images, test_labels = to_tensor(data['test_images'], data['test_labels'])
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

def get_dataloaders(dataset_name, batch_size):
    if dataset_name == 'FetalPlanesDB':
        train_files, train_labels, test_files, test_labels, label_map = load_fetal_planes_db('/datasets/Fetal-Planes-DB/dataset')
        train_dataset = GenericImageDataset(train_files, train_labels, mode="train")
        test_dataset = GenericImageDataset(test_files, test_labels, mode="val")
    elif dataset_name == 'KvasirV2':
        train_files, train_labels, test_files, test_labels, label_map = load_kvasir_v2('/datasets/kvasir-v2')
        train_dataset = GenericImageDataset(train_files, train_labels, mode="train")
        test_dataset = GenericImageDataset(test_files, test_labels, mode="val")
    elif dataset_name == 'CPN_X-ray':
        train_files, train_labels, test_files, test_labels, label_map = load_kvasir_v2('/datasets/dvntn9yhd2-1')
        train_dataset = GenericImageDataset(train_files, train_labels, mode="train")
        test_dataset = GenericImageDataset(test_files, test_labels, mode="val")
    else:
        train_images, train_labels, val_images, val_labels, test_images, test_labels = load_medmnist_npz(f'/datasets/{dataset_name.lower()}_224.npz')
        train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
        test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
