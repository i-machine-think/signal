import os

import numpy as np

import torchvision.transforms
import torch.utils.data as data

from enums.dataset_type import DatasetType
from helpers.file_helper import FileHelper


class MessageDataset(data.Dataset):
    def __init__(self, unique_name: str, dataset_type: DatasetType):
        super().__init__()

        self._file_helper = FileHelper()
        self._messages = self._get_message_data(unique_name, dataset_type)

        indices = self._get_indices_data(unique_name, dataset_type)
        self._raw_data = np.load(self._file_helper.get_input_path(dataset_type))[
            indices
        ]

        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToPILImage(), torchvision.transforms.ToTensor()]
        )

    def __getitem__(self, index):
        message = self._messages[index, :]
        raw_data = self._raw_data[index, :]

        raw_data = self.transforms(raw_data)

        return message, raw_data

    def __len__(self):
        return len(self._messages)

    def _get_message_data(self, unique_name, dataset_type):
        messages_filename = f"{unique_name}.{dataset_type}.messages.npy"
        messages_data = np.load(
            os.path.join(self._file_helper.messages_folder_path, messages_filename)
        )

        return messages_data

    def _get_indices_data(self, unique_name, dataset_type):
        indices_filename = f"{unique_name}.{dataset_type}.indices.npy"
        indices_data = np.load(
            os.path.join(self._file_helper.messages_folder_path, indices_filename)
        )

        return indices_data
