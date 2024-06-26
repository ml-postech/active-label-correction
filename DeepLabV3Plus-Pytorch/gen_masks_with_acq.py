import os
import json
import numpy as np
from PIL import Image

name = "Pascal"

if name == "Pascal":
    label_root_path = '/hdd/hdd4/khy/Revisiting-Pascal/deeplab/datasets/pascal_voc_seg/SegmentationClassRaw/'
    spx_root_path = '/hdd/hdd4/khy/Grounded-Segment-Anything/outputs/0.2/obj_jpg/'
    n_label_root_path = '/hdd/hdd4/khy/Grounded-Segment-Anything/outputs/0.2/mask_jpg/'

    imageset_path = '/data/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
    with open(imageset_path, 'r') as f:
        lines = f.readlines()
    image_list = [x.strip() for x in lines]

    imageset_path = '/data/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
    with open(imageset_path, 'r') as f:
        lines = f.readlines()
    val_list = [x.strip() for x in lines]

    image_name_path = {}
    for image_name in image_list:
        label_path = os.path.join(label_root_path, image_name + '.png')
        obj_path = os.path.join(spx_root_path, image_name + '.png') 
        n_label_path = os.path.join(n_label_root_path, image_name + '.png')
        image_name_path[image_name] = [label_path, obj_path, n_label_path]

names = [
    # 'sim_dic_0.0_.pkl',
    # 'cil_dic_0.0_.pkl',
    # 'softmin_dic_0.0_.pkl',
]
for name in names:
    # init_0
    parts = name.split('.')
    save_root_path = '/hdd/hdd4/khy/DeepLabV3Plus-Pytorch/masks/voc_pixelpick/5000/' + parts[0] + parts[1]
    os.makedirs(save_root_path, exist_ok=True)

    pixel_path = '/hdd/hdd4/khy/DeepLabV3Plus-Pytorch/acquisition/init_0/ppick_cil_dic.pkl'    
    pixel_dict = np.load(pixel_path, allow_pickle=True)

    pkl_path = '/hdd/hdd4/khy/DeepLabV3Plus-Pytorch/acquisition/init_0/' + name
    pkl_dict = np.load(pkl_path, allow_pickle=True)
    if 'aiou' in name:
        sorted_dict = dict(sorted(pkl_dict.items(), key=lambda item: item[1]))
    else:
        sorted_dict = dict(sorted(pkl_dict.items(), key=lambda item: -item[1][0]))

    count = 0
    pkl_images = {}
    for idx, key in enumerate(sorted_dict.keys()):
        if count == 5000: break

        image_name, obj = key
        if 'aiou' in name and obj == 0:
            continue 

        if image_name not in pkl_images.keys():
            pkl_images[image_name] = [obj]
        else:
            pkl_images[image_name].append(obj)
        count += 1

    print("Train")
    for idx, image_name in enumerate(image_name_path.keys()):
        print(name, idx, image_name)
        label_path, obj_path, n_label_path = image_name_path[image_name]
        labels = np.array(Image.open(label_path))
        objects = np.array(Image.open(obj_path))
        n_labels = np.array(Image.open(n_label_path))

        if image_name in pkl_images.keys():
            for obj in pkl_images[image_name]:
                key = (image_name, obj)
                _, rep_x, rep_y = pixel_dict[key]
                rep_label = labels[rep_x, rep_y]

                if 'aiou' in name:
                    obj_x, obj_y = np.where(objects == obj)
                    relabel_x = obj_x
                    relabel_y = obj_y
                else:
                    _, relabel_x, relabel_y = sorted_dict[key]

                n_region_labels = n_labels[relabel_x, relabel_y]
                region_labels = labels[relabel_x, relabel_y]
                bin = (n_region_labels == region_labels).astype('int')
                acc = np.sum(bin) / bin.shape[0]
                if acc <= 0.5:
                    n_labels[relabel_x, relabel_y] = rep_label

        final_label = Image.fromarray(n_labels.astype('uint8'))
        final_label.save(save_root_path + '/' + image_name + '.png')
        
    print("Valid")
    for idx, image_name in enumerate(val_list):
        print(name, idx, image_name)
        label_path = os.path.join(label_root_path, image_name + '.png')
        labels = np.array(Image.open(label_path))
        final_label = Image.fromarray(labels.astype('uint8'))
        final_label.save(save_root_path + '/' + image_name + '.png')