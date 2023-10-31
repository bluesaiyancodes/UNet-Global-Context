import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CarSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_ids = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_ids[idx] + '.png')
        mask_name = os.path.join(self.mask_dir, self.image_ids[idx] + '.png')

        image = Image.open(img_name).convert("L")
        mask = Image.open(mask_name).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
