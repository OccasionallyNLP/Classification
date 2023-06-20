# python test.py --output_dir ../output/RM/point --test_data ../data/RM/dev.jsonl --batch_size 1 --check_point_dir ../output/RM/point --max_length 512 --eval_rank_type point

# python test.py --output_dir ../output/RM/list --test_data ../data/RM/dev.jsonl --batch_size 1 --check_point_dir ../output/RM/list --max_length 512 --eval_rank_type list

# python test.py --output_dir ../output/RM/regression --test_data ../data/RM/dev.jsonl --batch_size 1 --check_point_dir ../output/RM/regression --max_length 512 --eval_rank_type regression

python test.py --output_dir ../output/RM/pair --test_data ../data/RM/dev.jsonl --batch_size 1 --check_point_dir ../output/RM/pair --max_length 512 --eval_rank_type pair

