# dataset.py 내 CelebADataset 수정 예시

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import numpy as np

class CelebADataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, image_ids=None, attributes=None):
        self.df = pd.read_csv(csv_path)
        self.df.rename(columns={self.df.columns[0]: 'image_id'}, inplace=True)
        self.img_dir = img_dir
        self.transform = transform

        if image_ids is not None:
            self.df = self.df[self.df['image_id'].isin(image_ids)]

        if attributes is None:
            self.attributes = self.df.columns[1:].tolist()  # 전체 40개
        else:
            self.attributes = attributes  # 6개로 제한

        # -1 → 0 변환
        for attr in self.attributes:
            self.df[attr] = self.df[attr].apply(lambda x: 1 if x == 1 else 0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[self.attributes].values.astype(np.float32))
        return image, labels

    def __len__(self):
        return len(self.df)