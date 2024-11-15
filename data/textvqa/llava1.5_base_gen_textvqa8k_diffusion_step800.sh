torchrun --nproc_per_node 4 --master_port 29502 generate_with_aug.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image_file_list ../step1/textvqa_image_question_list_8k.json \
    --image_path train_images/ \
    --save_dir ./ \
    --res_file "textvqa_answer_file_8k_diffusion_step800.jsonl" \
    --augmentation "diffusion" \
    --noise_step 800 \


