from torch.utils.data import Dataset
import lmdb
from io import BytesIO
from PIL import Image
import torchvision.transforms as tfs
import os

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.images= os.listdir(path)
        self.train_paths = [path +_ for _ in self.images]

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.train_paths)

    def __getitem__(self, index):
        
        img_name = self.train_paths[index]

        # buffer = BytesIO(img_bytes)
        img = Image.open(img_name)
        img=img.resize((256,256))
        img = self.transform(img)

        return img


################################################################################

def get_celeba_dataset(data_root, config):
    train_transform = tfs.Compose([tfs.ToTensor(),
                                   tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                 inplace=True)])

    test_transform = tfs.Compose([tfs.ToTensor(),
                                  tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                inplace=True)])

    train_dataset = MultiResolutionDataset(os.path.join(data_root, 'train/'),
                                           train_transform, config.data.image_size)
    test_dataset = MultiResolutionDataset(os.path.join(data_root, 'test/'),
                                          test_transform, config.data.image_size)


    return train_dataset, test_dataset



