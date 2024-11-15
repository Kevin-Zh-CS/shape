CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node 3 --master_port 29502 textvqa/answer_ensemble.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image_file_list textvqa/textvqa_answer_file_8k_merged.jsonl \
    --image_path textvqa/train_images/ \
    --save_dir ./ \
    --res_file "textvqa/textvqa_answer_file_8k_dpo.jsonl" \