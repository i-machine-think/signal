import numpy as np
import torch
import random
from torch.utils.data.sampler import Sampler
import torchvision.transforms
from PIL import Image


class ShapesDataset:
    def __init__(
        self,
        features,
        mean=None,
        std=None,
        metadata=False,
        raw=False,
        dataset=None,
        validation_set=False,
    ):
        self.metadata = metadata
        self.raw = raw
        self.features = features
        self.dataset = dataset
        self.validation_set = validation_set

        if type(self.features) == type({}):
            self.keys = list(features.keys())

        if dataset is not None:
            pass

        if mean is None and type(features) == type({}):

            imgs = np.asarray([features[key].data for key in features])
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
            [torchvision.transforms.ToPILImage(), torchvision.transforms.ToTensor()]
        )

    def __getitem__(self, indices):
        if type(self.features) == type({}):
            target_key = self.keys[indices[0]]
            target_img = self.features[target_key].data
            list_key = list(target_key)
            lkey = list_key[5]

            distractors = []

            if self.raw:  # and not self.validation_set:
                target_img = self.transforms(target_img)

            # if self.validation_set:
            #     print(target_key)

            # print(len(distractors))
            return (target_img, distractors, indices, lkey)

            # return (target_img, distractors, indices, lkey)
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

        if self.raw:  # and not self.validation_set:
            target_img = self.transforms(target_img)

        return (target_img, distractors, indices, 0)

    def __len__(self):
        if type(self.features) == type({}):
            print("Dataset size is", len(self.features))
            return len(self.features)
        else:
            print("Dataset size is", self.features.shape[0])
            return self.features.shape[0]
