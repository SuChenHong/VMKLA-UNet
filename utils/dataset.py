from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os 
import h5py
import random
import torch
import torchvision.transforms as transforms
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

train_transformer = None
test_transformer = None

class Polyp_datasets(Dataset):
    def __init__(self, path, config, mode='train', test_datasets='CVC-300'):
        super(Polyp_datasets, self)
        self.mode = mode
        if mode == 'train':
            images = sorted(
                os.listdir(os.path.join(path, 'train/images'))
            )
            labels = sorted(
                os.listdir(os.path.join(path, 'train/masks'))
            )
            self.data = []
            for i in range(len(images)):
                img_path = os.path.join(path, 'train/images', images[i])
                label_path = os.path.join(path, 'train/masks', labels[i])
                self.data.append([img_path, label_path])
            self.transformer = config.train_transformer

        elif mode == 'val':
            images = sorted(
                os.listdir(os.path.join(path, f'val/{test_datasets}/images'))
            )
            labels = sorted(
                os.listdir(os.path.join(path, f'val/{test_datasets}/masks'))
            )
            self.data = []
            for i in range(len(images)):
                img_path = os.path.join(path, f'val/{test_datasets}/images', images[i])
                label_path = os.path.join(path, f'val/{test_datasets}/masks', labels[i])
                self.data.append([img_path, label_path])
            self.transformer = config.test_transformer 

    def __getitem__(self, index):
        img_path, label_path = self.data[index]
        img = np.array(Image.open(img_path).convert('RGB'))
        label = np.expand_dims(np.array(Image.open(label_path).convert('L')), axis=2) / 255
        img, label = self.transformer((img, label))
        return img, label

    def __len__(self):
        return len(self.data)
    

class NPY_datasets(Dataset):
    def __init__(self, path, config, mode='train'):
        super(NPY_datasets, self)
        if mode == 'train':
            images = sorted(
                os.listdir(os.path.join(path, 'train/images'))
            )
            labels = sorted(
                os.listdir(os.path.join(path, 'train/masks'))
            )
            self.data = []
            for i in range(len(images)):
                img_path = os.path.join(path, 'train/images', images[i])
                label_path = os.path.join(path, 'train/masks', labels[i])
                self.data.append([img_path, label_path])
            self.transformer = config.train_transformer
        elif mode == 'val':
            images = sorted(
                os.listdir(os.path.join(path, 'val/images'))
            )
            labels = sorted(
                os.listdir(os.path.join(path, 'val/masks'))
            )
            self.data = []
            for i in range(len(images)):
                img_path = os.path.join(path, 'val/images', images[i])
                label_path = os.path.join(path, 'val/masks', labels[i])
                self.data.append([img_path, label_path])
            self.transformer = config.test_transformer

        elif mode == 'test':
            images = sorted(
                os.listdir(os.path.join(path, 'test/images'))
            )
            labels = sorted(
                os.listdir(os.path.join(path, 'test/masks'))
            )
            self.data = []
            for i in range(len(images)):
                img_path = os.path.join(path, 'test/images', images[i])
                label_path = os.path.join(path, 'test/masks', labels[i])
                self.data.append([img_path, label_path])
            self.transformer = config.test_transformer
        else:
            raise ValueError('mode must be train, val or test')
    
    def __getitem__(self, index):
        img_path, label_path = self.data[index]
        img = np.array(Image.open(img_path).convert('RGB'))
        label = np.expand_dims(np.array(Image.open(label_path).convert('L')), axis=2) / 255
        img, label = self.transformer((img, label))
        return img, label
    
    def __len__(self):
        return len(self.data)


class Synapse_datasets(Dataset):
    
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

