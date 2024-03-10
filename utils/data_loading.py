import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.path_hyperparameter import ph
import os


class BasicDataset(Dataset):
    """ Basic dataset for train, evaluation and test.
    
    Attributes:
        t1_images_dir(str): path of t1 images.
        t2_images_dir(str): path of t2 images.
        labels_dir(str): path of labels.
        train(bool): ensure creating a train dataset or other dataset.
        t1_ids(list): name list of t1 images.
        t2_ids(list): name list of t2 images.
        train_transforms_all(class): data augmentation applied to t1 image, t2 image and label.
        train_transforms_image(class): noise addition only applied to t1 image and t2 image.
        t1_normalize(class): normalizer applied to t1 image.
        t2_normalize(class): normalizer applied to t2 image.
        to_tensor(class): convert array to tensor.

    """

    def __init__(self, t1_images_dir: str, t2_images_dir: str, labels_dir: str, train: bool,
                 t1_mean: list, t1_std: list, t2_mean: list, t2_std: list):
        """ Init of basic dataset.
        
        Parameter:
            t1_images_dir(str): path of t1 images.
            t2_images_dir(str): path of t2 images.
            labels_dir(str): path of labels.
            train(bool): ensure creating a train dataset or other dataset.
            t1_mean(list): t1 images std in three channel(RGB)
            t1_std(list): t1 images std in three channel(RGB)
            t2_mean(list): t2 images std in three channel(RGB)
            t2_std(list): t2 images std in three channel(RGB)

        """

        self.t1_images_dir = Path(t1_images_dir)
        self.t2_images_dir = Path(t2_images_dir)
        self.labels_dir = Path(labels_dir)
        self.train = train
        if self.train:
            txt_path = os.path.join(os.path.dirname(self.t1_images_dir), 'train.txt')
        else:
            txt_path = os.path.join(os.path.dirname(self.t1_images_dir), 'val.txt')
        # image name without suffix
        self.t1_ids = open(txt_path, 'r').readlines()
        logging.info(f'Creating dataset with {len(self.t1_ids)} examples')

        self.train_transforms_all = A.Compose([
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.Rotate(45, p=0.3),
            A.ShiftScaleRotate(p=0.3),
        ], additional_targets={'image1': 'image'})

        self.train_transforms_image = A.Compose(
            [A.OneOf([
                A.GaussNoise(p=1),
                A.HueSaturationValue(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.Emboss(p=1),
                A.MotionBlur(p=1),
            ], p=ph.noise_p)],
            additional_targets={'image1': 'image'})

        self.t1_normalize = A.Compose([
            A.Normalize(
                mean=t1_mean,
                std=t1_std)
        ])

        self.t2_normalize = A.Compose([
            A.Normalize(
                mean=t2_mean,
                std=t2_std)
        ])

        self.to_tensor = A.Compose([
            ToTensorV2()
        ], additional_targets={'image1': 'image'})

    def __len__(self):
        """ Return length of dataset."""

        return len(self.t1_ids)

    @classmethod
    def label_preprocess(cls, label):
        """ Binaryzation label."""

        label = np.where(label > 0, 1, 0)
        return label

    @classmethod
    def load(cls, filename):
        """Open image and convert image to array."""

        img = Image.open(filename)
        img = np.array(img)

        return img

    def __getitem__(self, idx):
        """ Index dataset.

        Index image name list to get image name, search image in image path with its name,
        open image and convert it to array.

        Preprocess array, apply data augmentation and noise addition(optional) on it,
        random exchange t1 and t2 array, and convert array to tensor.

        Parameter:
            idx(int): index of dataset.

        Return:
            t1_tensor(tensor): tensor of t1 image.
            t2_tensor(tensor): tensor of t2 image.
            label_tensor(tensor): tensor of label.
            name(str): the same name of t1 image, t2 image and label.
        """

        data_info = self.t1_ids[idx].strip().split('  ')
        t1_img_file, t2_img_file, label_file = data_info

        t1_img = self.load(t1_img_file)
        t2_img = self.load(t2_img_file)

        label = self.load(label_file)
        label = self.label_preprocess(label)

        if self.train:
            sample = self.train_transforms_all(image=t1_img, image1=t2_img, mask=label)
            t1_img, t2_img, label = sample['image'], sample['image1'], sample['mask']
            sample = self.train_transforms_image(image=t1_img, image1=t2_img)
            t1_img, t2_img = sample['image'], sample['image1']

        t1_img = self.t1_normalize(image=t1_img)['image']
        t2_img = self.t2_normalize(image=t2_img)['image']
        if self.train:
            # random exchange t1_img and t2_img
            if random.choice([0, 1]):
                t1_img, t2_img = t2_img, t1_img
        sample = self.to_tensor(image=t1_img, image1=t2_img, mask=label)
        # ipdb.set_trace()
        t1_tensor, t2_tensor, label_tensor = sample['image'].contiguous(),\
                                             sample['image1'].contiguous(), sample['mask'].contiguous()
        name = t1_img_file

        return t1_tensor, t2_tensor, label_tensor, name
