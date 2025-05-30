import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import yaml
import matplotlib.pyplot as plt

class YOLOEmotionDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load class names from data.yaml
        yaml_path = os.path.join(root_dir, 'data.yaml')
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            self.class_names = config.get('names', ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
        except Exception as e:
            print(f"Warning: Could not load data.yaml: {e}")
            self.class_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.label_dir = os.path.join(root_dir, split, 'labels')
        
        # Get image files with error handling
        try:
            self.image_files = [f for f in os.listdir(self.image_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        except FileNotFoundError:
            print(f"Error: Image directory not found: {self.image_dir}")
            self.image_files = []
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Read and convert to grayscale
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image {img_path}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            # Return a blank image placeholder
            gray = np.zeros((96, 96), dtype=np.uint8)
            return self.transform(gray) if self.transform else gray, 0
        
        # Read label file
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        class_id = 0  # Default class
        try:
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found: {label_path}")
                
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Use first annotation (assuming one face per image)
            if not lines:
                raise ValueError("Empty label file")
                
            line = lines[0].strip().split()
            class_id = int(line[0])
            
            # Convert YOLO bbox to pixel coordinates
            h, w = gray.shape
            cx, cy, bw, bh = map(float, line[1:5])
            x1 = max(0, int((cx - bw/2) * w))
            y1 = max(0, int((cy - bh/2) * h))
            x2 = min(w, int((cx + bw/2) * w))
            y2 = min(h, int((cy + bh/2) * h))
            
            # Crop face
            face = gray[y1:y2, x1:x2]
            
            if face.size == 0:
                print(f"Warning: Empty face crop in {img_name}")
                face = gray  # Fallback to entire image
            
            # Resize while maintaining aspect ratio
            h, w = face.shape
            if h < 10 or w < 10:  # Face too small
                face = gray  # Fallback to entire image
                h, w = face.shape
            
            # Pad to square if needed
            if h != w:
                size = max(h, w)
                pad_h = (size - h) // 2
                pad_w = (size - w) // 2
                face = np.pad(face, ((pad_h, size-h-pad_h), (pad_w, size-w-pad_w)), 
                             mode='constant', constant_values=0)
            
            # Resize to 96x96
            face = cv2.resize(face, (96, 96))
            
        except Exception as e:
            print(f"Error processing {label_path}: {e}")
            # Use entire image as fallback
            face = cv2.resize(gray, (96, 96))
        
        # Apply transformations
        if self.transform:
            try:
                face = self.transform(face)
            except Exception as e:
                print(f"Error transforming image {img_name}: {e}")
                # Create a placeholder tensor
                placeholder = torch.zeros(1, 96, 96)
                return placeholder, class_id
        
        return face, class_id

def get_transforms():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3)
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, val_transform

def visualize_sample(dataset, index, save_path=None):
    """Visualize a sample from the dataset"""
    image, label = dataset[index]
    
    if isinstance(image, torch.Tensor):
        image = image.numpy().squeeze()
    elif isinstance(image, np.ndarray):
        image = image.squeeze()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(f"Class: {dataset.class_names[label]}")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    dataset = YOLOEmotionDataset(root_dir='../data', split='train')
    print(f"Dataset size: {len(dataset)}")
    
    # Visualize some samples
    os.makedirs("../reports/samples", exist_ok=True)
    for i in [0, 100, 500, 1000]:
        if i < len(dataset):
            save_path = f"../reports/samples/sample_{i}.png"
            visualize_sample(dataset, i, save_path)
            print(f"Saved sample {i} to {save_path}")
    
    # Print dataset statistics
    print("\nDataset statistics:")
    print(f"Number of samples: {len(dataset)}")
    print(f"Image shape: {dataset[0][0].shape if len(dataset) > 0 else 'N/A'}")