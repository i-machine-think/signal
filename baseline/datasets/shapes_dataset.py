import numpy as np
import torch
import random
from torch.utils.data.sampler import Sampler
import torchvision.transforms
from PIL import Image


class ShapesDataset:
    def __init__(
        self, features, mean=None, std=None, metadata=False, raw=False, dataset=None
    ):
        self.metadata = metadata
        self.raw = raw
        self.features = features

        self.obverter_setup = False
        self.dataset = dataset
        if dataset is not None:
            self.obverter_setup = True

        if mean is None:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std[np.nonzero(std == 0.0)] = 1.0  # nan is because of dividing by zero
        self.mean = mean
        self.std = std

        if not raw and not metadata:
            self.features = (features - self.mean) / (2 * self.std)

        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor()
            ]
        )

    def __getitem__(self, indices):
        target_idx = indices[0]
        distractors_idxs = indices[1:]

        distractors = []
        for d_idx in distractors_idxs:
            distractor_img = self.features[d_idx]
            if self.raw:
                distractor_img = self.transforms(distractor_img)
            distractors.append(distractor_img)

        target_img = self.features[target_idx]
        if self.raw:
            target_img = self.transforms(target_img)

        return (target_img, distractors, indices)

    def __len__(self):
        if self.obverter_setup:
            return self.dataset.shape[0]
        else:
            return self.features.shape[0]


class ImagesSampler(Sampler):
    def __init__(self, data_source, k, shuffle):
        self.n = len(data_source)
        self.k = k
        self.shuffle = shuffle
        assert self.k < self.n

    def __iter__(self):
        indices = []

        if self.shuffle:
            targets = torch.randperm(self.n).tolist()
        else:
            targets = list(range(self.n))

        for t in targets:
            arr = np.zeros(self.k + 1, dtype=int)  # distractors + target
            arr[0] = t
            distractors = random.sample(range(self.n), self.k)
            while t in distractors:
                distractors = random.sample(range(self.n), self.k)
            arr[1:] = np.array(distractors)

            indices.append(arr)

        return iter(indices)

    def __len__(self):
        return self.n
