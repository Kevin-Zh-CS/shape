CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node 3 --master_port 29502 ocrvqa/answer_ensemble.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image_file_list ocrvqa/ocrvqa_answer_file_8k_merged.jsonl \
    --image_path ocrvqa/images/ \
    --save_dir ./ \
    --res_file "ocrvqa/ocrvqa_answer_file_8k_dpo.jsonl" \