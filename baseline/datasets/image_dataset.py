from PIL import Image
import torchvision.transforms
import torch.utils.data as data


class ImageDataset(data.Dataset):
    def __init__(self, images):
        super().__init__()

        self.data = images
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # Normalize to (-1, 1)
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, index):
        image = self.data[index, :, :, :]
        image = self.transforms(image)
        return image

    def __len__(self):
        return self.data.shape[0]
