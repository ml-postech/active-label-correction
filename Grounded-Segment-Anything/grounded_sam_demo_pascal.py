import argparse
from math import e
import os
import copy
import token

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


num_dic = {
    '18440' : 'aeroplane',
    '11751' : 'aeroplane',
    '10165' : 'bicycle',
    '4743' : 'bird',
    '4049' : 'boat',
    '5835' : 'bottle',
    '3902' : 'bus',
    '2482' : 'car',
    '4937' : 'cat',
    '3242' : 'chair',
    '11190' : 'cow',
    '7759' : 'diningtable',
    '10880' : 'diningtable',
    '3899' : 'dog',
    '3586' : 'horse',
    '5013' : 'motorbike',
    '5638' : 'motorbike',
    '3489' : 'motorbike',
    '2711' : 'person',
    '8962' : 'pottedplant',
    '3064' : 'pottedplant',
    '24759' : 'pottedplant',
    '4630' : 'pottedplant',
    '8351' : 'sheep',
    '10682' : 'sofa',
    '3345' : 'train',
    '2694' : 'tvmonitor',
    '8202' : 'tvmonitor',
    '15660' : 'tvmonitor'
}


label_dic = {
    "background" : 0, "aeroplane" : 1, "bicycle" : 2, "bird" : 3, "boat" : 4, "bottle" : 5,
    "bus" : 6, "car" : 7, "cat" : 8, "chair" : 9, "cow" : 10, "diningtable" : 11, "dog" : 12,
    "horse" : 13, "motorbike" : 14, "person" : 15, "pottedplant" : 16, "sheep" : 17,
    "sofa" : 18, "train" : 19, "tvmonitor" : 20, "ignore" : 255
}


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt = logits_filt[filt_mask]  # num_filt, 256

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        idx = torch.argmax(logit)
        pred_phrase = tokenized['input_ids'][idx]
        pred_phrase = num_dic[str(pred_phrase)]
        # pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(args, image_name, output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_dic = {}
    n, _, _, _ = mask_list.shape
    for i in range(n):
        mask = mask_list[i,:,:,:].cpu().numpy().astype('int')
        mask_dic[i] = np.sum(mask)

    obj_img = torch.zeros(mask_list.shape[-2:])
    mask_img = torch.zeros(mask_list.shape[-2:])
    sorted_dict = dict(sorted(mask_dic.items(), key=lambda item: -item[1]))
    for idx in sorted_dict.keys():
        mask = mask_list[idx,:,:,:].cpu().numpy()[0]
        label = label_list[idx]
        name, logit = label.split('(')

        label_val = label_dic[name]
        mask_img[mask == True] = label_val
        obj_img[mask == True] = idx + 1

    final_label = Image.fromarray(mask_img.numpy().astype('uint8'))
    final_label.save(output_dir + "/" + str(args.box_threshold) + "/mask_jpg/" + image_name + ".png")
    
    final_obj = Image.fromarray(obj_img.numpy().astype('uint8'))
    final_obj.save(output_dir + "/" + str(args.box_threshold) + "/obj_jpg/" + image_name + ".png")

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, str(args.box_threshold) + "/mask_json/" + image_name + ".json"), 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", help="using sam-hq for prediction")
    parser.add_argument("--input_image", type=str, default="temp", help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # initialize SAM
    model = load_model(config_file, grounded_checkpoint, device=device)
    model = model.to(device)
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    devkit_path = '/data/datasets/VOCdevkit/'
    image_root_path = devkit_path + 'VOC2012/JPEGImages'
    imageset_path = devkit_path + 'VOC2012/ImageSets/Segmentation/train.txt'
    with open(imageset_path, 'r') as f:
         lines = f.readlines()
    image_list = [x.strip() for x in lines]

    image_name_path = {}
    for image_name in image_list:
        image_path = os.path.join(image_root_path, image_name + '.jpg')
        image_name_path[image_name] = image_path

    os.makedirs(os.path.join(output_dir, str(args.box_threshold) + "/mask_json/"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, str(args.box_threshold) + "/mask_jpg/"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, str(args.box_threshold) + "/obj_jpg/"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, str(args.box_threshold) + "/sam_jpg/"), exist_ok=True)

    for idx, image_name in enumerate(image_name_path.keys()):
        print(idx, image_name)
        if os.path.exists(os.path.join(output_dir, str(args.box_threshold) + "/mask_json/" + image_name + ".json")):
            continue

        image_path = image_name_path[image_name]
        image_pil, image = load_image(image_path)
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
        if transformed_boxes.shape[0] == 0:
            mask_np = np.zeros((H, W)) # background
            final_label = Image.fromarray(mask_np.astype('uint8'))
            final_label.save(output_dir + "/" + str(args.box_threshold) + "/mask_jpg/" + image_name + ".png")

            final_obj = Image.fromarray(mask_np.astype('uint8'))
            final_obj.save(output_dir + "/" + str(args.box_threshold) + "/obj_jpg/" + image_name + ".png")
            continue

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        
        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)
        plt.axis('off')
        plt.savefig(
            os.path.join(output_dir, str(args.box_threshold) + "/sam_jpg/" + image_name + ".jpg"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )

        save_mask_data(args, image_name, output_dir, masks, boxes_filt, pred_phrases)