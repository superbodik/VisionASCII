from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class VisionASCIIDataset(Dataset):
    def __init__(self, image_dir, ascii_dir, transform=None):
        self.image_dir = image_dir
        self.ascii_dir = ascii_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.ascii_files = sorted(os.listdir(ascii_dir))

    def __len__(self):
        return min(len(self.image_files), len(self.ascii_files))

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        ascii_path = os.path.join(self.ascii_dir, self.ascii_files[idx])

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        with open(ascii_path, "r", encoding="utf-8") as f:
            ascii_str = f.read()

        ascii_tensor = torch.tensor([ord(c) for c in ascii_str if c != '\n'], dtype=torch.long)

        return image, ascii_tensor
