for init_seed in 1; do
    python main.py --model deeplabv3plus_resnet101 \
        --gpu_id 4 \
        --year 2012 \
        --crop_val \
        --lr 0.01 \
        --crop_size 513 \
        --batch_size 16 \
        --random_seed $init_seed \
        --data_root /data/datasets \
        --dataset vocpsim00 \
        --mask_dir /hdd/hdd4/khy/DeepLabV3Plus-Pytorch/masks/init_0/5000/sim_dic_00_/
done;