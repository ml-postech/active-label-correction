import os
import utils
import random
import pickle
import network
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt


from PIL import Image
from tqdm import tqdm
from torch.utils import data
from scipy.stats import entropy
from scipy.special import softmax


from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics


def get_argparser():
    parser = argparse.ArgumentParser()

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name]))
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--ckpt", default='./checkpoints/best_deeplabv3plus_resnet101_vocrandseeds15k_2.pth', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='2',
                        help="GPU ID")
    return parser


def transform_train(sample, opts):
    train_transform = et.ExtCompose([
        # et.ExtResize(opts.crop_size),
        et.ExtCenterCrop(opts.crop_size),
        et.ExtToTensor(),
        et.ExtNormalize(mean=opts.mean, std=opts.std),
    ])
    image, label = train_transform(sample['image'], sample['label'])
    sample = {'image': image, 'label': label}
    return sample


def acquisition(model, args):
    model.eval()
    devkit_path = '/data/datasets/VOCdevkit/'
    image_root_path = devkit_path + 'VOC2012/JPEGImages'

    label_root_path = '/hdd/hdd4/khy/Grounded-Segment-Anything/outputs/0.2/mask_jpg/'
    spx_root_path = '/hdd/hdd4/khy/Grounded-Segment-Anything/outputs/0.2/obj_jpg/'
    
    save_path = '/hdd/hdd2/khy/icml24/soft_label/temp/'
    os.makedirs(save_path, exist_ok=True)

    imageset_path = devkit_path + 'VOC2012/ImageSets/Segmentation/train.txt'
    with open(imageset_path, 'r') as f:
         lines = f.readlines()
    image_list = [x.strip() for x in lines]

    image_name_path = {}
    for image_name in image_list:
        image_path = os.path.join(image_root_path, image_name + '.jpg')
        label_path = os.path.join(label_root_path, image_name + '.png')
        obj_path = os.path.join(spx_root_path, image_name + '.png')
        image_name_path[image_name] = [image_path, label_path, obj_path]

    for num, image_name in enumerate(image_name_path.keys()):
        print(num, image_name)
        image_path, label_path, _ = image_name_path[image_name]
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)
     
        h, w = label.size
        y0 = int(round((args.crop_size - h) / 2.))
        x0 = int(round((args.crop_size - w) / 2.))

        sample = {'image': image, 'label': label}
        sample = transform_train(sample, args)
        sample["image"] = sample["image"].reshape(1, 3, args.crop_size, args.crop_size) # y, x

        prob = model(sample["image"].cuda())
        prob_np = np.array(prob.detach().cpu())
        prob_np = np.squeeze(prob_np)
        prob_np = prob_np[:, x0:(x0+w), y0:(y0+h)]
        prob_soft = softmax(prob_np, axis=0)

        with open(save_path + image_name + '.pkl', 'wb') as file:
            pickle.dump(prob_soft, file)

def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 21
    opts.mean = [0.485, 0.456, 0.406]
    opts.std = [0.229, 0.224, 0.225]
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    del checkpoint

    acquisition(model, opts)


if __name__ == '__main__':
    main()
