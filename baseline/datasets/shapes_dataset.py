import numpy as np
import torch
import random
from torch.utils.data.sampler import Sampler
import torchvision.transforms
from PIL import Image


class ShapesDataset:
    def __init__(
        self, features, mean=None, std=None, metadata=False, raw=False, dataset=None, step3_distractors = None, validation_set = False
    ):
        self.metadata = metadata
        self.raw = raw
        self.features = features

        self.obverter_setup = False
        self.dataset = dataset
        self.step3_distractors = step3_distractors
        self.validation_set = validation_set

        if type(self.features) == type({}):
            self.keys = list(features.keys())

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
            target_key = self.keys[indices[0]]
            target_img = self.features[target_key].data
            list_key = list(target_key)
            lkey = list_key[5]

            distractors = []

            # distractors = self.step3_distractors[target_key]
            # for i, d in enumerate(distractors):
            #     distractors[i] = self.transforms(d.data)
            for distractor_img in self.step3_distractors[target_key]:
                # distractor_img = self.step3_distractors[target_key]
                if self.raw:
                    distractor_img = self.transforms(distractor_img.data)
                distractors.append(distractor_img)
                if len(self.step3_distractors[target_key]) == 1:
                    distractors.append(distractor_img)

            if self.raw:# and not self.validation_set:
                target_img = self.transforms(target_img)

            # if self.validation_set:
            #     print(target_key)

            # print(len(distractors))
            return (target_img, distractors, indices, lkey)

                # print('train',len(distractors))
            # else:
            #     target_img = []
            #     for i in range(5):
            #         list_key[5] = str(i)
            #         class_key = ''.join(list_key)
            #         # class_key = target_key#'111111ab'
            #         # if i < 4:
            #         #     class_key = self.keys[indices[i]]
            #         # target_key = self.keys[np.random.randint(max(indices))]
            #         # list_key = list(target_key)
            #         # list_key[5] = str(np.random.randint(0,5))
            #         # target_key = ''.join(list_key)

            #         # class_key = target_key
            #         # if i == 2:
            #         #     class_key = target_key # testing related, remove when done
            #         class_distractors = []
            #         for distractor_img in self.step3_distractors[class_key]:
            #             if self.raw:
            #                 distractor_img = self.transforms(distractor_img.data)
            #             class_distractors.append(distractor_img)
            #         distractors.append(class_distractors)

            #         targ = self.features[target_key].data
            #         if self.raw:
            #             targ = self.transforms(targ)
            #         target_img.append(targ)

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

        if self.raw:# and not self.validation_set:
            target_img = self.transforms(target_img)

        return (target_img, distractors, indices, 0)

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