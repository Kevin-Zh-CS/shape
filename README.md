
## SHAPE : Self-Improved Visual Preference Alignment by Iterative Generation

### Install

**Create a conda environment and install packages**

```
conda create -n seva python==3.10 -y
conda activate shape
pip install torch==2.0.1 torchvision==0.15.2
pip install -e .
```

### Dataset

**We expect the ****image dataset** to have the following structure:

```
data/
|-- texvqa/
|---- train_images/
......
|-- ocrvqa/
|---- images/
```

**You can download the image on the official website.**

### Training

**You need to first download weights of **[LLaVA-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b).

**For running with DPO:**

```
MODEL_VERSION=llava-shape-dpo-1.5

OCR_DPO_DATA=data/step2/ocrvqa/ocrvqa_answer_file_8k_dpo.jsonl
TEXT_DPO_DATA=data/step2/textvqa/textvqa_answer_file_8k_dpo.jsonl

deepspeed --include localhost:0,1,2,3 seva/train_dpo_ours.py \
    --deepspeed seva/scripts/zero3.json \
    --lora_enable True --lora_r 1024 --lora_alpha 2048 --mm_projector_lr 0 \
    --deepspeed seva/scripts/zero3_offload.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --ocr_data_path ${OCR_DPO_DATA} \
    --ocr_image_path data/step2/ocrvqa/images/ \
    --textvqa_data_path ${TEXT_DPO_DATA} \
    --textvqa_image_path data/step2/textvqa/train_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/${MODEL_VERSION} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 12000 \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${MODEL_VERSION} \
    --beta 0.1



```

### Evaluation

**Refer to **[LLaVa-1.5](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) for a comprehension evaluation of multiple Benchmarks.
