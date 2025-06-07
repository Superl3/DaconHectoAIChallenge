import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []

        if is_test:
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    img_path = os.path.join(root_dir, fname)
                    self.samples.append((img_path,))
        else:
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                if not os.path.isdir(cls_folder):
                    continue
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        img_path = os.path.join(cls_folder, fname)
                        label = self.class_to_idx[cls_name]
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            img_path = self.samples[idx][0]
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            if self.transform:
                transformed_data = self.transform(image=image)
                image = transformed_data['image']
            return image
        else:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            if self.transform:
                transformed_data = self.transform(image=image)
                image = transformed_data['image']
            return image, label
