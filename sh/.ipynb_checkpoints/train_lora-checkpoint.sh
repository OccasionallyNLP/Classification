python train_lora.py --output_dir D:/jupyter_notebook/output/classification/nli/lora/kogpt/base --train_data D:/jupyter_notebook/data/NLI/dev.jsonl --val_data D:/jupyter_notebook/data/NLI/dev.jsonl --logging_term 1000 --epochs 5 --eval_epoch 1 --batch_size 1 --warmup 50 --ptm_path skt/kogpt2-base-v2 --max_length 512 --early_stop True --early_stop_metric acc --early_stop_metric_is_max_better True --save_model_every_epoch True --patience 3 --lr 2e-5 --fp16 False --fp16_model True --accumulation_steps 1 --n_labels 3 