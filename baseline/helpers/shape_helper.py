import os
import pickle
import numpy as np
from random import shuffle

import torch

from .image_helper import get_image
from .feature_helper import get_features
from .file_helper import FileHelper

from enums.dataset_type import DatasetType

SEED = 42

file_helper = FileHelper()


def generate_image_dataset(size, seed=42):
    """
    Generates an image dataset using the seed passed
    """
    images = []
    for i in range(size):
        images.append(get_image(seed + i))
    shuffle(images)
    return images


def get_image_datasets(train_size, valid_size, test_size, seed=42):
    """
    Returns split image dataset with the desired sizes for train/valid/test
    """
    data = generate_image_dataset(
        train_size + valid_size + test_size, seed=seed)
    train_data = data[:train_size]
    valid_data = data[train_size: train_size + valid_size]
    test_data = data[train_size + valid_size:]
    assert len(train_data) == train_size
    assert len(valid_data) == valid_size
    assert len(test_data) == test_size
    return train_data, valid_data, test_data


def generate_shapes_dataset():
    """
    Generates shapes dataset and extract features
    @TODO - add parameters to extend generation and feature extraction process
    """

    np.random.seed(SEED)

    # From Serhii's original experiment
    train_size = 74504
    valid_size = 8279
    test_size = 40504

    # --- Generate Datasets ----
    train_data, valid_data, test_data = get_image_datasets(
        train_size, valid_size, test_size, seed=SEED
    )

    sets = {DatasetType.Train: train_data,
            DatasetType.Valid: valid_data, DatasetType.Test: test_data}

    # --- Save Generated Datasets ----
    for dataset_type, dataset_data in sets.items():
        set_inputs = np.asarray([image.data[:, :, 0:3]
                                 for image in dataset_data])
        np.save(file_helper.get_input_path(dataset_type), set_inputs)
        set_metadata = [image.metadata for image in dataset_data]
        pickle.dump(set_metadata, open(
            file_helper.get_metadata_path(dataset_type), "wb"))


if __name__ == "__main__":
    generate_shapes_dataset()
