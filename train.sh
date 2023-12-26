CUDA_VISIBLE_DEVICES=4,5,6,7 python train_sharding.py \
            --num_gpus 4 \
            --seed 1 \
            --lr 0.00025 \
            --clip_grad 1 \
            --arch base \
            --norm_last_layer \
            --data_path /shared/j.jang/pathai/data/TCGA-lung-patches-256 \
            --output_dir /shared/js.yun/DINO_sagemaker \
            --process_list_csv /shared/j.jang/pathai/data/TCGA-lung-patches-256/process_list_autogen.csv \
            --split_csv /shared/j.jang/pathai/data/TCGA-lung-luad+lusc-TMB-323-splits-seed100/task_1_tumor_vs_normal_100/splits_9.csv \
            --num_workers 4 \
            --num_epochs 100 \
            --pretrained_path /shared/j.jang/pathai/HIPT/2-Weakly-Supervised-Subtyping/lg_clips/medical/synth.pt