from torch.utils.data import Dataset
import torch
import torch.utils.data as Data
import scipy.io as scio
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from torchvision import transforms
import random
import os
import pdb
import numpy as np
from transform_my_mask_no_contour import transform_rotate, transform_translate_horizontal, transform_translate_vertical, transform_flip, transform_shear

class RandomGaussianBlur(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return img


def default_loader(path, IorM='rgb'):
    if IorM == 'rgb':
        return Image.open(path).convert('RGB')
    else:
        return Image.open(path).convert('L')


class MyDataset(Dataset):
    def __init__(self, img_label_txt, loader=default_loader, mode='test'):
        # print(imgtxt)
        img_label = []
        path = open(img_label_txt, 'r')
        for line in path:
            line = line.strip('\n')
            line = line.rstrip()
            img_label.append(line)

        self.img_label = img_label
        self.imgs_num = len(img_label)

        self.toTensor = transforms.ToTensor()
        self.HorizontalFlip = transforms.RandomHorizontalFlip(p=0.5)

        self.degrees = random.uniform(0, 10)
        # self.RandomAffine_degrees = transforms.RandomAffine(degrees=self.degrees)  # 转

        self.ColorJitter = transforms.ColorJitter(brightness=0.1)
        self.RandomGaussianBlur = RandomGaussianBlur()
        self.resize = transforms.Resize((512, 512))
        self.loader = loader
        self.mode = mode

        # self.RandomGaussianBlur = tr.RandomGaussianBlur()


    def __getitem__(self, index):

        # imgname = self.imgs[index]
        imglabel = self.img_label[index]

        temp = imglabel.strip().split('\t')
        # print(index)
        # print(temp)

        img = self.loader(temp[0], IorM='L')  # L

        # all bones mask
        label_mask = self.loader(temp[1], IorM='Binary')
        # clavicle mask
        labelClavicle_mask = self.loader(temp[2], IorM='Binary')
        # post rib mask
        labelPosteriorrib_mask = self.loader(temp[3], IorM='Binary')
        # pre rib mask
        labelPrerib_mask = self.loader(temp[4], IorM='Binary')


        mask = [label_mask, labelClavicle_mask, labelPosteriorrib_mask, labelPrerib_mask]
        # print(os.path.join(self.img_path[index//self.imgs_num], imgname))
        # print(os.path.join(self.label_path[0], imgname))

        if self.mode == 'train':
            if random.random() > 0.5:
                img, mask = transform_flip(img, mask)

            if random.random() < 0.25:
                img, mask = transform_translate_horizontal(img, mask, scale=random.uniform(0, 0.05))
            if random.random() < 0.25:
                img, mask = transform_translate_vertical(img, mask,  scale=random.uniform(0, 0.05))

            if random.random() < 0.25:
                img, mask = transform_rotate(img, mask)

            if random.random() < 0.25:
                img, mask = transform_shear(img, mask)

            if random.random() >= 0.6:
                img = self.ColorJitter(img)

            # img = self.RandomGaussianBlur(img)

        img = self.resize(img)
        img = self.toTensor(img)  # (1, 512, 512)

        for i in range(len(mask)):
            mask[i] = self.resize(mask[i])
            mask[i] = self.toTensor(mask[i])

        return img, mask, temp[0]

    def __len__(self):
        return len(self.img_label)



if __name__ == '__main__':
    img_label_txt = r''

    train_datasets = MyDataset(img_label_txt, mode='test')
    trainloader = Data.DataLoader(dataset=train_datasets, batch_size=1, shuffle=False, num_workers=0)

    for step, (imgs, mask, _) in enumerate(trainloader):
        # print(mask[0].shape)
        print(imgs.shape)




