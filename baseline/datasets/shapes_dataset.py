import numpy as np
import torch
import random
from torch.utils.data.sampler import Sampler
import torchvision.transforms
from PIL import Image


class ShapesDataset:
    def __init__(
        self, features, mean=None, std=None, metadata=False, raw=False, dataset=None, step3_distractors = None
    ):
        self.metadata = metadata
        self.raw = raw
        self.features = features

        self.obverter_setup = False
        self.dataset = dataset
        self.step3_distractors = step3_distractors

        if dataset is not None:
            self.obverter_setup = True

        if mean is None and type(features) == type({}):

            imgs = np.asarray([features[key].data
                                 for key in features])
            mean = np.mean(imgs, axis=0)
            std = np.std(imgs, axis=0)
            std[np.nonzero(std == 0.0)] = 1.0  # nan is because of dividing by zero

        elif mean is None:
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
        if type(self.features) == type({}):
            target_key = [k for k in self.features][indices[0]]

            distractors = []
            for distractor_img in self.step3_distractors[target_key]:
                if self.raw:
                    distractor_img = self.transforms(distractor_img.data)
                distractors.append(distractor_img)

            target_img = self.features[target_key].data
        else:
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
            if type(self.features) == type({}):
                print('Dataset size is',len(self.features))
                return len(self.features)
            else:
                print('Dataset size is',self.features.shape[0])
                return self.features.shape[0]