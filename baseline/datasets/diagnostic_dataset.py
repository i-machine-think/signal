import os

import numpy as np

import torchvision.transforms
import torch.utils.data as data

from enums.dataset_type import DatasetType
from helpers.file_helper import FileHelper
from helpers.metadata_helper import get_metadata_properties


class DiagnosticDataset(data.Dataset):
    def __init__(self, unique_name: str, dataset_type: DatasetType):
        super().__init__()

        self._unique_name = unique_name
        self._dataset_type = dataset_type

        self._file_helper = FileHelper()
        self._messages = self._get_message_data()

        indices = self._get_indices_data()
        self._properties = get_metadata_properties(dataset_type)[indices]

    def __getitem__(self, index):
        message = self._messages[index, :]
        properties = self._properties[index, :]
        return message, properties

    def __len__(self):
        return len(self._properties)

    def _get_message_data(self):
        messages_filename = f"{self._unique_name}.{self._dataset_type}.messages.npy"
        messages_data = np.load(
            os.path.join(self._file_helper.messages_folder_path, messages_filename)
        )

        return messages_data

    def _get_indices_data(self):
        indices_filename = f"{self._unique_name}.{self._dataset_type}.indices.npy"
        indices_data = np.load(
            os.path.join(self._file_helper.messages_folder_path, indices_filename)
        )

        return indices_data
