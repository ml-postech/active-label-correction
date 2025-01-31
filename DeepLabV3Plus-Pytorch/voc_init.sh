for init_seed in 1; do
    python main.py --model deeplabv3plus_resnet101 \
        --gpu_id 0,1,2,3,4,5,6,7 \
        --year 2012 \
        --crop_val \
        --lr 0.01 \
        --crop_size 513 \
        --batch_size 16 \
        --random_seed $init_seed \
        --data_root /data/datasets \
        --dataset voc \
        --mask_dir /hdd/hdd2/khy/temp/active-label-correction/Grounded-Segment-Anything/outputs_voc/0.2/mask_jpg/
done;