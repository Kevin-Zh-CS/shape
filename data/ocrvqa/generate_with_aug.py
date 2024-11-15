import os
import json
import tqdm
import torch
import argparse
import datetime
from PIL import Image
from transformers import TextStreamer
import torchvision.transforms as transforms
import sys
sys.path.insert(0, "../../src")


from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import random
from PIL import ImageFilter

def convert_dict_to_tensor(results, device):
    part_tensor = json.dumps(results)
    part_tensor = torch.Tensor([ord(part_tensor[i]) for i in range(len(part_tensor))]).long().to(device)
    return part_tensor


def add_diffusion_noise(image_tensor, noise_step=500):
    num_steps = 1000  # Number of diffusion steps
    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    
    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 
    return image_tensor_cd

def use_color_jitter(image_tensor):
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    return color_jitter(image_tensor)

def use_flip(image_tensor):
    flipper = transforms.RandomHorizontalFlip()
    return flipper(image_tensor)

def main(args):
    # 初始化模型
    disable_torch_init()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    model_name = get_model_name_from_path(args.model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=device
    )

    conv_mode = "llava_v1"

    # 获取问题文件
    image_file_list = open(args.image_file_list)
    lines = list(image_file_list.readlines())
    rank, word_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    step = len(lines) // word_size + 1
    # step = 1000
    start, end = rank * step, (rank + 1) * step
    print("start form " + str(start) + " and end with " + str(end))
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    res_file = os.path.join(save_dir, args.res_file)
    if rank == 0:
        print("Generating answers...")

    with open(res_file, "a") as result_file:
        for line in tqdm.tqdm(lines[start:end]):
            data = json.loads(line)
            message_input = data["text"]
            image_path = os.path.join(args.image_path, data["image"])
            image = Image.open(image_path).convert("RGB")
            conv = conv_templates[conv_mode].copy()
            roles = conv.roles if "mpt" not in model_name.lower() else ('user', 'assistant')
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            if args.augmentations:
                if "diffusion" in args.augmentations:
                    image_tensor = add_diffusion_noise(image_tensor, args.noise_step)
                elif "contrast" in args.augmentations:
                    image_tensor = use_color_jitter(image_tensor)
                elif "flip" in args.augmentations:
                    image_tensor = use_flip(image_tensor)

            inp = message_input
            if image is not None:
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(roles[0], inp)
            conv.append_message(roles[1], None)
            prompt = conv.get_prompt()


            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=1.0,
                    max_new_tokens=512,
                    stopping_criteria=[stopping_criteria])
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()


            result = {
                "question": message_input,
                "answer": outputs,
                "image_id": data['image'].split('.')[0]
            }
            result_file.write(json.dumps(result) + '\n')
            result_file.flush()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--res_file", type=str, default="generate.jsonl")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument('--augmentations', nargs='+', type=str, default=[])
    parser.add_argument("--noise_step", default=500, type=int)
    parser.add_argument("--image_file_list", default=None, type=str)
    parser.add_argument("--image_path", type=str, required=True)
    
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args.dist_backend = "nccl"
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["LOCAL_RANK"]),
        timeout=datetime.timedelta(days=365),
    )
    
    main(args)
