import pickle
import os
import numpy as np

from enums.dataset_type import DatasetType
from .file_helper import FileHelper

file_helper = FileHelper()

def one_hot(a):
    ncols = a.max() + 1
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out


def get_shapes_metadata(dataset=DatasetType.Test):
    """
    Args:
        dataset (str, opt) from {"train", "valid", "test"}
    returns one hot encoding of metada - compressed version of true concepts
    @TODO implement loading from file rather than loading each time
    """

    test_meta = pickle.load(
        open(file_helper.get_metadata_path(dataset), "rb")
    )

    compressed_test_images = np.zeros((len(test_meta), 5))
    for i, m in enumerate(test_meta):
        pos_h, pos_w = (np.array(m["shapes"]) != None).nonzero()
        pos_h, pos_w = pos_h[0], pos_w[0]
        color = m["colors"][pos_h][pos_w]
        shape = m["shapes"][pos_h][pos_w]
        size = m["sizes"][pos_h][pos_w]
        compressed_test_images[i] = np.array(
            [color, shape, size, pos_h, pos_w])
    compressed_test_images = compressed_test_images.astype(np.int)

    # return one hot encoding
    # note this will have 15 dimensions and not 14 as expected
    one_hot_derivations = one_hot(compressed_test_images).reshape(
        compressed_test_images.shape[0], -1
    )
    return one_hot_derivations

if __name__ == "__main__":
    m = get_shapes_metadata()
    print(m.shape)