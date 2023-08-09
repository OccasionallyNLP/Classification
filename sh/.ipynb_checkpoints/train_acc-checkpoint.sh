# single gpu
CUDA_VISIBLE_DEVICES="0" accelerate launch train_accelerate.py --output_dir D:/jupyter_notebook/output/classification/nli/roberta/base --train_data D:/jupyter_notebook/data/NLI/dev.jsonl --val_data D:/jupyter_notebook/data/NLI/dev.jsonl --logging_term 1000 --epochs 2 --eval_epoch 1 --batch_size 4 --warmup 50 --ptm_path klue/roberta-base --max_length 512 --early_stop True --early_stop_metric acc --early_stop_metric_is_max_better True --save_model_every_epoch True --patience 3 --lr 2e-5 --fp16 True --accumulation_steps 128 --n_labels 3

# multi gpu
accelerate launch --multi_gpu --mixed_precision=fp16
