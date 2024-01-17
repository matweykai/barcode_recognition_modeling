import os
from typing import Union, Optional
import pandas as pd

import albumentations as albu
import cv2
from torch.utils.data import Dataset


TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


class BarCodeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_folder: str,
        transforms: Optional[TRANSFORM_TYPE] = None,
        return_path: bool = False,
    ):
        self.transforms = transforms

        self.crops = []
        self.codes = []
        self.return_path = return_path
        self.df = df

        for i in range(len(df)):
            image = cv2.imread(os.path.join(data_folder, df['filename'][i]))[..., ::-1]
            x1 = int(df['x_from'][i])
            y1 = int(df['y_from'][i])
            x2 = int(df['x_from'][i]) + int(df['width'][i])
            y2 = int(df['y_from'][i]) + int(df['height'][i])
            crop = image[y1:y2, x1:x2]

            if crop.shape[0] > crop.shape[1]:
                crop = cv2.rotate(crop, 2)

            self.crops.append(crop)
            self.codes.append(str(df['code'][i]))

    def __getitem__(self, idx):
        text = self.codes[idx]
        image = self.crops[idx]

        data = {
            'image': image,
            'text': text,
            'text_length': len(text),
        }

        if self.transforms:
            data = self.transforms(**data)

        if self.return_path:
            return data['image'], data['text'], data['text_length'], self.df['filename'][idx]
        else:
            return data['image'], data['text'], data['text_length']

    def __len__(self):
        return len(self.crops)
