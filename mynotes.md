https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md

laion-400m is a 400M image text dataset

Download the metadata
wget -l1 -r --no-parent https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/
mv the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/ .
Download the images with img2dataset
img2dataset --url_list laion400m-meta --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder laion400m-data --processes_count 16 --thread_count 128 --image_size 256\
             --save_additional_columns '["NSFW","similarity","LICENSE"]' --enable_wandb True
img2dataset --url_list laion400m-meta --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder laion400m-data --processes_count 32 --thread_count 128 --image_size 256\
             --save_additional_columns '["NSFW","similarity","LICENSE"]' --enable_wandb False
Benchmark
This can be downloaded at 1300 sample/s so it takes 3.5 days to download with one 16 cores 2Gbps machine. The result is 10TB


srun --export ALL --pty -p gpu --gres=gpu:8 --cpus-per-task=40 --mem=256G --time=2400:00:00 bash
srun --export ALL --pty -p unlimited --cpus-per-task=8 --mem=16G --time=2400:00:00 bash

# have not been tested. use it at your own discretion
# the original experiment was run on tpu v3-256.
# this example script assumes 8 gpus, each with huge memory. Tune batchsize, warmup, and lr accordingly if you have different machine setups.
# precision: (choose from 'amp', 'amp_bf16', 'amp_bfloat16', 'bf16', 'fp16', 'pure_bf16', 'pure_fp16', 'fp32')   

LOG_DIR=./logs/
rm -rf ${LOG_DIR}
torchrun --nproc_per_node 8 -m training.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --train-data '' \
    --train-num-samples 100000 \
    --dataset-type synthetic \
    --lr "2.048e-3" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 3200 \
    --wd 0.2 \
    --batch-size 2048 \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --epochs 1 \
    --workers 4 \
    --model RN50 \
    --precision 'fp16' \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --force-image-size 84 \
    --grad-checkpointing \
    --log-every-n-steps 32 \
    --seed 0 \
    --logs ${LOG_DIR} \
    --imagenet-val '/scratch/local/zhongz2/imagenet'


LOG_DIR=/data/zhongz2/temp15/openclip_logs/
python -m training.main \
    --dataset-type "csv" \
    --save-frequency 1 \
    --train-data="/lscratch/$SLURM_JOB_ID/train.csv"  \
    --val-data="/lscratch/$SLURM_JOB_ID/val.csv"  \
    --csv-img-key image \
    --csv-caption-key caption \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --warmup 500 \
    --batch-size=4 \
    --lr=1e-5 \
    --wd=0.2 \
    --epochs=10 \
    --workers=8 \
    --model ViT-B-32 \
    --seed 0 \
    --logs ${LOG_DIR} \
    --save-most-recent


LOG_DIR=/data/zhongz2/temp15/openclip_logs/
python -m training.main \
    --dataset-type "csv" \
    --save-frequency 1 \
    --train-data="/lscratch/$SLURM_JOB_ID/train.csv"  \
    --val-data="/lscratch/$SLURM_JOB_ID/val.csv"  \
    --csv-img-key image \
    --csv-caption-key caption \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --warmup 500 \
    --batch-size=4 \
    --lr=5e-5 \
    --wd=0.2 \
    --epochs=12 \
    --workers=8 \
    --model ViT-B-32 \
    --seed 0 \
    --logs ${LOG_DIR} \
    --save-most-recent






# using openai pretrained

GPUID=0
LOG_DIR=/data/zhongz2/temp15/openclip_logs_using_pretrained/
CUDA_VISIBLE_DEVICES=${GPUID} python -m training.main \
    --dataset-type "csv" \
    --save-frequency 1 \
    --train-data="/lscratch/$SLURM_JOB_ID/train.csv"  \
    --val-data="/lscratch/$SLURM_JOB_ID/val.csv"  \
    --csv-img-key image \
    --csv-caption-key caption \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --warmup 500 \
    --batch-size 4 \
    --lr 1e-5 \
    --wd 0.2 \
    --epochs 20 \
    --workers 8 \
    --model ViT-B-32 \
    --pretrained "openai" \
    --seed 0 \
    --logs ${LOG_DIR} \
    --save-most-recent

GPUID=1
LR=5e-5
LOG_DIR=/data/zhongz2/temp15/openclip_logs_using_pretrained/
CUDA_VISIBLE_DEVICES=${GPUID} python -m training.main \
    --dataset-type "csv" \
    --save-frequency 1 \
    --train-data="/lscratch/$SLURM_JOB_ID/train.csv"  \
    --val-data="/lscratch/$SLURM_JOB_ID/val.csv"  \
    --csv-img-key image \
    --csv-caption-key caption \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --warmup 500 \
    --batch-size 4 \
    --lr ${LR} \
    --wd 0.2 \
    --epochs 20 \
    --workers 8 \
    --model ViT-B-32 \
    --pretrained "openai" \
    --seed 0 \
    --logs ${LOG_DIR} \
    --save-most-recent



# using openai pretrained, changed learning rate

GPUID=0
LR=1e-6
LOG_DIR=/data/zhongz2/temp15/openclip_logs_using_pretrained_lr${LR}/
CUDA_VISIBLE_DEVICES=${GPUID} python -m training.main \
    --dataset-type "csv" \
    --save-frequency 1 \
    --train-data="/lscratch/$SLURM_JOB_ID/train.csv"  \
    --val-data="/lscratch/$SLURM_JOB_ID/val.csv"  \
    --csv-img-key image \
    --csv-caption-key caption \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --warmup 500 \
    --batch-size 4 \
    --lr ${LR} \
    --wd 0.2 \
    --epochs 20 \
    --workers 8 \
    --model ViT-B-32 \
    --pretrained "openai" \
    --seed 0 \
    --logs ${LOG_DIR} \
    --save-most-recent

GPUID=1
LR=5e-6
LOG_DIR=/data/zhongz2/temp15/openclip_logs_using_pretrained_lr${LR}/
CUDA_VISIBLE_DEVICES=${GPUID} python -m training.main \
    --dataset-type "csv" \
    --save-frequency 1 \
    --train-data="/lscratch/$SLURM_JOB_ID/train.csv"  \
    --val-data="/lscratch/$SLURM_JOB_ID/val.csv"  \
    --csv-img-key image \
    --csv-caption-key caption \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --warmup 500 \
    --batch-size 4 \
    --lr ${LR} \
    --wd 0.2 \
    --epochs 20 \
    --workers 8 \
    --model ViT-B-32 \
    --pretrained "openai" \
    --seed 0 \
    --logs ${LOG_DIR} \
    --save-most-recent




GPUID=0
LR=1e-5
LOG_DIR=/data/zhongz2/temp15/openclip_logs_using_pretrained_lr${LR}_20240118/
CUDA_VISIBLE_DEVICES=${GPUID} python -m training.main \
    --dataset-type "csv" \
    --save-frequency 1 \
    --train-data="/lscratch/$SLURM_JOB_ID/train1.csv"  \
    --val-data="/lscratch/$SLURM_JOB_ID/val1.csv"  \
    --csv-img-key image \
    --csv-caption-key caption \
    --aug-cfg scale='(0.7, 1.2)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.5 gray_scale_prob=0.2 \
    --warmup 500 \
    --batch-size 16 \
    --lr ${LR} \
    --beta1 0.9 \
    --beta2 0.95 \
    --wd 0.2 \
    --epochs 20 \
    --workers 8 \
    --model ViT-B-32 \
    --pretrained "openai" \
    --seed 0 \
    --logs ${LOG_DIR} \
    --save-most-recent


GPUID=1
LR=5e-4
LOG_DIR=/data/zhongz2/temp15/openclip_logs_using_pretrained_lr${LR}_20240119/
CUDA_VISIBLE_DEVICES=${GPUID} python -m training.main \
    --dataset-type "csv" \
    --save-frequency 1 \
    --train-data="/lscratch/$SLURM_JOB_ID/train1.csv"  \
    --val-data="/lscratch/$SLURM_JOB_ID/val1.csv"  \
    --csv-img-key image \
    --csv-caption-key caption \
    --aug-cfg scale='(0.7, 1.2)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.5 gray_scale_prob=0.2 \
    --warmup 500 \
    --batch-size 16 \
    --lr ${LR} \
    --beta1 0.9 \
    --beta2 0.95 \
    --wd 0.2 \
    --epochs 20 \
    --workers 8 \
    --model ViT-B-32 \
    --pretrained "openai" \
    --seed 0 \
    --logs ${LOG_DIR} \
    --save-most-recent






