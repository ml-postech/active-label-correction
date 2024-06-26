import random
from skimage.io import imread

import torch
from torch.utils import data
import torchvision.transforms.functional as TF


class Kvasir(data.Dataset):
    def __init__(
        self,
        input_paths: list,
        target_paths: list,
        transform_input=None,
        transform_target=None,
        hflip=False,
        vflip=False,
        affine=False,
    ):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.hflip = hflip
        self.vflip = vflip
        self.affine = affine

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        input_ID = self.input_paths[index]
        target_ID = self.target_paths[index]

        x, y = imread(input_ID), imread(target_ID)
        if y.ndim == 2:
            ### duplicate channel into three
            y = y[:, :, None].repeat(3, axis=2)

        x = self.transform_input(x)
        y = self.transform_target(y)

        if self.hflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)

        if self.vflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)

        if self.affine:
            angle = random.uniform(-180.0, 180.0)
            h_trans = random.uniform(-352 / 8, 352 / 8)
            v_trans = random.uniform(-352 / 8, 352 / 8)
            scale = random.uniform(0.5, 1.5)
            shear = random.uniform(-22.5, 22.5)
            x = TF.affine(x, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
            y = TF.affine(y, angle, (h_trans, v_trans), scale, shear, fill=0.0)
        y = (y > 0.5)
        return x.float(), y.float()



# import numpy as np
# import torch
# import torch.utils.data as data
# from torchvision import datasets, transforms
# import os
# from PIL import Image, ImageOps

# class Kvasir(data.Dataset):

#     def __init__(self, root=None, split='train', cross='1', transform=None, mask_dir=None):
#         #self.h_image_size, self.w_image_size = image_size[0], image_size[1]
#         self.split = split
#         self.transform = transform
#         self.cross = cross

#         self.item_image = np.load(root + "datamodel/{}_data_{}.npy".format(self.split, self.cross))

#         if mask_dir == None:
#             self.item_gt = np.load(root + "datamodel/{}_label_{}.npy".format(self.split, self.cross))
#         else:
#             pass

#         print(np.bincount(self.item_gt.flatten()))

#     def __getitem__(self, index):
#         items_im = self.item_image
#         items_gt = self.item_gt
#         img_name = items_im[index]
#         label_name = items_gt[index]
#         label_name = np.where(label_name>200, 1, 0)

#         image = Image.fromarray(np.uint8(img_name))
#         mask = Image.fromarray(np.uint8(label_name))

#         #mask = np.eye(2)[mask]

#         if self.transform:
#             image, mask = self.transform(image, mask)

#         return image, mask

#     def __len__(self):
#         return len(self.item_image)