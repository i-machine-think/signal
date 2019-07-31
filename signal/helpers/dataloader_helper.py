import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from .metadata_helper import get_shapes_metadata
from .shape_helper import generate_shapes_dataset
from .feature_helper import get_features
from .file_helper import FileHelper

from datasets.shapes_dataset import ShapesDataset
from samplers.images_sampler import ImagesSampler

from enums.dataset_type import DatasetType

file_helper = FileHelper()


def get_shapes_features(device, dataset=DatasetType.Valid, mode="features"):
    """
    Returns numpy array with matching features
    Args:
        dataset (str) in {'train', 'valid', 'test'}
        mode (str) in {"features", "raw"}
    """
    if mode == "features":
        features_path = file_helper.get_features_path(dataset)

        if not os.path.isfile(features_path):
            images = np.load(file_helper.get_input_path(dataset))

            features = get_features(images, device)
            np.save(features_path, features)
            assert len(features) == len(images)

        return np.load(features_path)
    else:
        images = np.load(file_helper.get_input_path(dataset))
        return images


def get_dataloaders(
    device, batch_size=16, k=3, debug=False, dataset="all", dataset_type="features"
):
    """
    Returns dataloader for the train/valid/test datasets
    Args:
        batch_size: batch size to be used in the dataloader
        k: number of distractors to be used in training
        debug (bool, optional): whether to use a much smaller subset of train data
        dataset (str, optional): whether to return a specific dataset or all
                                 options are {"train", "valid", "test", "all"}
                                 default: "all"
        dataset_type (str, optional): what datatype encoding to use: {"meta", "features", "raw"}
                                      default: "features"
    """
    if dataset_type == "raw":
        train_features = np.load(file_helper.train_input_path)
        valid_features = np.load(file_helper.valid_input_path)
        test_features = np.load(file_helper.test_input_path)

        train_dataset = ShapesDataset(train_features, raw=True)

        # All features are normalized with train mean and std
        valid_dataset = ShapesDataset(
            valid_features, mean=train_dataset.mean, std=train_dataset.std, raw=True
        )

        test_dataset = ShapesDataset(
            test_features, mean=train_dataset.mean, std=train_dataset.std, raw=True
        )

    if dataset_type == "features":

        train_features = get_shapes_features(device, dataset=DatasetType.Train)
        valid_features = get_shapes_features(device, dataset=DatasetType.Valid)
        test_features = get_shapes_features(device, dataset=DatasetType.Test)

        if debug:
            train_features = train_features[:10000]

        train_dataset = ShapesDataset(train_features)

        # All features are normalized with train mean and std
        valid_dataset = ShapesDataset(
            valid_features, mean=train_dataset.mean, std=train_dataset.std
        )

        test_dataset = ShapesDataset(
            test_features, mean=train_dataset.mean, std=train_dataset.std
        )

    if dataset_type == "meta":
        train_meta = get_shapes_metadata(dataset=DatasetType.Train)
        valid_meta = get_shapes_metadata(dataset=DatasetType.Valid)
        test_meta = get_shapes_metadata(dataset=DatasetType.Test)

        train_dataset = ShapesDataset(train_meta.astype(np.float32), metadata=True)
        valid_dataset = ShapesDataset(valid_meta.astype(np.float32), metadata=True)
        test_dataset = ShapesDataset(test_meta.astype(np.float32), metadata=True)

    train_data = DataLoader(
        train_dataset,
        pin_memory=True,
        batch_sampler=BatchSampler(
            ImagesSampler(train_dataset, k, shuffle=True),
            batch_size=batch_size,
            drop_last=False,
        ),
    )

    valid_data = DataLoader(
        valid_dataset,
        pin_memory=True,
        batch_sampler=BatchSampler(
            ImagesSampler(valid_dataset, k, shuffle=False),
            batch_size=batch_size,
            drop_last=False,
        ),
    )

    test_data = DataLoader(
        test_dataset,
        pin_memory=True,
        batch_sampler=BatchSampler(
            ImagesSampler(test_dataset, k, shuffle=False),
            batch_size=batch_size,
            drop_last=False,
        ),
    )

    if dataset == "train":
        return train_data
    if dataset == "valid":
        return valid_data
    if dataset == "test":
        return test_data
    else:
        return train_data, valid_data, test_data


def get_shapes_dataloader(
    device, batch_size=16, k=3, debug=False, dataset="all", dataset_type="features"
):
    """
    Args:
        batch_size (int, opt): batch size of dataloaders
        k (int, opt): number of distractors
    """

    if not os.path.exists(file_helper.train_features_path):
        print("Features files not present - generating dataset")
        generate_shapes_dataset()

    return get_dataloaders(
        device,
        batch_size=batch_size,
        k=k,
        debug=debug,
        dataset=dataset,
        dataset_type=dataset_type,
    )
