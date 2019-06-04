from PIL import Image
import torchvision.transforms
import torch.utils.data as data


class ImageDataset(data.Dataset):
    def __init__(self, images):
        super().__init__()

        self.data = images

        H, W = images.shape[1], images.shape[2]
        if H != 128 or W != 128:
            self.transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(),
                    # Resize
                    torchvision.transforms.Resize((128, 128), Image.LINEAR),
                    torchvision.transforms.ToTensor(),
                    # Normalize to (-1, 1)
                    torchvision.transforms.Normalize(
                        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    # Normalize to (-1, 1)
                    torchvision.transforms.Normalize(
                        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

    def __getitem__(self, index):
        image = self.data[index, :, :, :]
        image = self.transforms(image)
        return image

    def __len__(self):
        return self.data.shape[0]
