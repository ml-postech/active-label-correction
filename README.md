# Active Label Correction for Semantic Segmentation with Foundation Models
This repository is the official implementation of ["Active Label Correction for Semantic Segmentation with Foundation Models"](https://arxiv.org/abs/2403.10820) accepted by ICML 2024. The projection page is available [here](https://cskhy16.github.io/alc/).

## Abstract
Training and validating models for semantic segmentation require datasets with pixel-wise annotations, which are notoriously labor-intensive. Although useful priors such as foundation models or crowdsourced datasets are available, they are error-prone. We hence propose an effective framework of active label correction (ALC) based on a design of correction query to rectify pseudo labels of pixels, which in turn is more annotator-friendly than the standard one inquiring to classify a pixel directly according to our theoretical analysis and user study. Specifically, leveraging foundation models providing useful zero-shot predictions on pseudo labels and superpixels, our method comprises two key techniques: (i) an annotator-friendly design of correction query with the pseudo labels, and (ii) an acquisition function looking ahead label expansions based on the superpixels. Experimental results on PASCAL, Cityscapes, and Kvasir-SEG datasets demonstrate the effectiveness of our ALC framework, outperforming prior methods for active semantic segmentation and label correction. Notably, utilizing our method, we obtained a revised dataset of PASCAL by rectifying errors in 2.6 million pixels in PASCAL dataset.

## Initial Dataset Preparation
Our codes operate on ["Grounded-Segment-Anything"](https://github.com/IDEA-Research/Grounded-Segment-Anything). We solve the problem of multi-classes in object detection by giving each object the most likely as a pseudo-label, i.e., we just ignore the text threshold.

## Active Label Correction
Our codes are based on ["DeepLabV3Plus-Pytorch"](https://github.com/VainF/DeepLabV3Plus-Pytorch). Active learning generally follows these steps: (1) training the model (main.py, voc.sh), (2) selecting samples through acquisition (soft_label.py, acq_on_sam.py), and (3) labeling process (gen_masks_with_acq.py).

## Cite
Please cite our paper if you use the model or this code in your own work:
```
@inproceedings{kim2024active,
  title={Active Label Correction for Semantic Segmentation with Foundation Models},
  author={Hoyoung Kim and Sehyun Hwang and Suha Kwak and Jungseul Ok},
  booktitle=ICML,
  year={2024},
  url={https://arxiv.org/abs/2403.10820}
}
```
