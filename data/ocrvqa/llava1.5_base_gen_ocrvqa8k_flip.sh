CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node 3 --master_port 29502 generate_with_aug.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image_file_list ../step1/ocrvqa_image_question_list_8k.json \
    --image_path images/ \
    --save_dir ./ \
    --res_file "ocrvqa_answer_file_8k_flip.jsonl" \
    --augmentation "flip"



