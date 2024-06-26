import os
import json
import pickle
import numpy as np
import multiprocessing as mp

from PIL import Image
from scipy.spatial import distance
from scipy.special import softmax
from scipy.spatial.distance import cosine


# Thres, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
thres = 0.0
soft_label_path = '/hdd/hdd2/khy/icml24/soft_label/vocrandpsam6k2/'
spx_root_path = '/hdd/hdd4/khy/Grounded-Segment-Anything/outputs/0.2/obj_jpg/'
n_label_root_path = '/hdd/hdd4/khy/DeepLabV3Plus-Pytorch/masks/voc_rand_p_sam_2/6000/sim_dic_00_/'

imageset_path = '/data/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
with open(imageset_path, 'r') as f:
    lines = f.readlines()
image_list = [x.strip() for x in lines]

image_name_path = {}
for image_name in image_list:
    obj_path = os.path.join(spx_root_path, image_name + '.png') 
    n_label_path = os.path.join(n_label_root_path, image_name + '.png')
    soft_path = os.path.join(soft_label_path, image_name + '.pkl')
    image_name_path[image_name] = [obj_path, n_label_path, soft_path]


def cos_sim(a, b):
    c, n = a.shape

    cos = []
    for i in range(n):
        each_a = a[:, i]
        cos.append(1 - cosine(each_a, b)) # 1 => same
    cos = np.array(cos)
    return cos


def acq(image_name):
    cil_pixel, cil_dic, sim_dic = {}, {}, {}

    obj_path, n_label_path, pseudo_path = image_name_path[image_name]
    objects = np.array(Image.open(obj_path))
    n_labels = np.array(Image.open(n_label_path))
    soft_labels = np.load(pseudo_path, allow_pickle=True) # soft label
    arg_labels = np.argmax(soft_labels, axis=0) # hard label
    h, w = np.shape(n_labels)

    for obj in np.unique(objects):
        if obj == 0:
            continue
        obj_x, obj_y = np.where(objects == obj)
        region_labels = arg_labels[obj_x, obj_y]
        pseudo_doms, counts = np.unique(region_labels, return_counts=True)
        pseudo_dom = pseudo_doms[np.argmax(counts)]

        ids = np.where(region_labels == pseudo_dom)
        obj_px, obj_py = obj_x[ids], obj_y[ids]
        region_feats = soft_labels[:, obj_px, obj_py]
        mean_feat = np.mean(region_feats, axis=1)
        similarity = cos_sim(region_feats, mean_feat)
        sim_id = np.argmax(similarity)

        # Representative pixel
        rep_x, rep_y = obj_px[sim_id], obj_py[sim_id]  
        rep_feat = soft_labels[:, rep_x, rep_y]
        rep_label = n_labels[rep_x, rep_y]
        if rep_label == 255:
            rep_label = arg_labels[rep_x, rep_y]

        # Larger, noiseness
        key = (image_name, obj)
        pixel_val = 1 - soft_labels[rep_label, rep_x, rep_y]
        cil_pixel[key] = (pixel_val, rep_x, rep_y)

        # Label expansion
        region_feats = soft_labels[:, obj_x, obj_y]
        cos_array = cos_sim(region_feats, rep_feat)
        cos_ids = np.where(cos_array >= thres)
        cos_idx, cos_idy = obj_x[cos_ids], obj_y[cos_ids]
        length = cos_idx.shape[0]

        cil_arr = []
        for each_x, each_y in zip(cos_idx, cos_idy):
            each_label = n_labels[each_x, each_y]
            if each_label == 255:
                each_label = arg_labels[each_x, each_y]
            each_cil = 1 - soft_labels[each_label, each_x, each_y]
            cil_arr.append(each_cil) 
        cil_arr = np.array(cil_arr) # s => 1 - s

        cil_alpha = np.array([1 / length for i in range(length)])
        sim_alpha = np.array(cos_array[cos_ids])

        cil = np.sum(cil_arr * cil_alpha)
        cil_dic[key] = (cil, cos_idx, cos_idy)
        sim = np.sum(cil_arr * sim_alpha)
        sim_dic[key] = (sim, cos_idx, cos_idy)

        save_path = '/hdd/hdd2/khy/icml24/acquisition/vocrandpsam6k2/' + image_name + '/'
        os.makedirs(save_path, exist_ok=True)
        with open(save_path + 'cil_pixel.pkl', 'wb') as file:
            pickle.dump(cil_pixel, file)
        with open(save_path + 'cil_dic_' + str(thres) + '_.pkl', 'wb') as file:
            pickle.dump(cil_dic, file)
        with open(save_path + 'sim_dic_' + str(thres) + '_.pkl', 'wb') as file:
            pickle.dump(sim_dic, file)


if __name__ == '__main__':
    keys = list(image_name_path.keys())

    max_cpu_num = 40
    process = []
    for idx, image_name in enumerate(keys):
        print(idx, image_name)
        p = mp.Process(target=acq, args=(image_name,))
        p.start()
        process.append(p)

        if len(process) >= max_cpu_num:
            process[0].join()
            process.pop(0)