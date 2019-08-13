import numpy as np
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from ..datasets.image_dataset import ImageDataset
from .file_helper import FileHelper

BATCH_SIZE = 16  # batch size used to extract features

file_helper = FileHelper()


def get_features(images, device):
    print("Extracting features")

    if not os.path.isfile(file_helper.model_checkpoint_path):
        return ValueError(
            "Feature Extractor is missing. Train baseline using 'raw' features."
        )

    model = torch.load(
        file_helper.model_checkpoint_path,
        map_location=lambda storage, location: storage,
    )

    model.to(device)

    dataloader = DataLoader(ImageDataset(images), batch_size=BATCH_SIZE)

    features = []
    for i, x in tqdm(enumerate(dataloader), total=len(dataloader)):
        x = x.to(device)
        y = model(x)
        y = y.view(y.size(0), -1).detach().cpu().numpy()
        features.append(y)

    # concatenate all features
    features = np.concatenate(features, axis=0)
    return features
