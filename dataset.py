import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image


class RotatedDataset(Dataset):
    def __init__(self, csv_file, root_path, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_path = Path(root_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.root_path / self.data.iloc[index, 0]
        image = read_image(img_path.as_posix())
        image = image / 255.0

        if self.transform:
            image = self.transform(image)

        angle = torch.tensor(float(self.data.iloc[index, 1]))
        digit = torch.tensor(int(self.data.iloc[index, 2]))

        return image, angle, digit
