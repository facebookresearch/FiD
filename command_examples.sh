# NGPU=8 python -m torch.distributed.launch --nproc_per_node=8 train_reader.py```

# CUDA_VISIBLE_DEVICES=0,1 NGPU=2 python -m torch.distributed.launch --nproc_per_node=2 train_reader.py```

```
NGPU=<num of gpus in one node> python -m torch.distributed.launch --nproc_per_node=<num of gpus in one node> train_reader.py \
        --train_data train_data.json \
        --eval_data eval_data.json \
        --model_size base \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name my_experiment \
        --checkpoint_dir checkpoint \
        --use_checkpoint \
        --lr 0.00005 \
        --optim adamw \
        --scheduler linear \
        --weight_decay 0.01 \
        --text_maxlength 250 \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --total_step 15000 \
        --warmup_step 1000 \
```