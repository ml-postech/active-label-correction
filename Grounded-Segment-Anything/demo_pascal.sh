export CUDA_VISIBLE_DEVICES=0

python grounded_sam_demo_pascal.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --output_dir "outputs_voc" \
  --box_threshold 0.1 \
  --text_threshold 0.25 \
  --text_prompt "Aeroplane. Bicycle. Bird. Boat. Bottle. Bus. Car. Cat. Chair. Cow. Diningtable. Dog. Horse. Motorbike. Person. Pottedplant. Sheep. Sofa. Train. Tvmonitor." \
  --device "cuda"