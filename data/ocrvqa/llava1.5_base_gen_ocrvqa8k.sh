torchrun --nproc_per_node 3 --master_port 29502 generate_with_aug.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image_file_list ../step1/ocrvqa_image_question_list_8k.json \
    --image_path images/ \
    --save_dir ./ \
    --res_file_1 "ocrvqa_answer_file_8k_base_1.jsonl" \
    --res_file_2 "ocrvqa_answer_file_8k_base_2.jsonl"


