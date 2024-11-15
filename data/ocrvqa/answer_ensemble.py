import os
import json
import tqdm
import torch
import argparse
import datetime
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import torchvision.transforms as transforms
import sys
sys.path.insert(0, "../../src")


from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


import torchvision.transforms as transforms
import random
from PIL import ImageFilter

def convert_dict_to_tensor(results, device):
    part_tensor = json.dumps(results)
    part_tensor = torch.Tensor([ord(part_tensor[i]) for i in range(len(part_tensor))]).long().to(device)
    return part_tensor


def main(args):
    # Model
    disable_torch_init()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    model_name = get_model_name_from_path(args.model_path)
    if args.model_base is None:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path, 
            model_base=args.model_base, 
            model_name=model_name,
            device=device
        )
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base, 
            model_name="llava_lora_model",
            device=device
        )

    conv_mode = "llava_v1"

    ## get question file
    image_file_list = open(args.image_file_list)
    lines = list(image_file_list.readlines())
    rank, word_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    step = len(lines) // word_size + 1
    start, end = rank * step, (rank + 1) * step
    results = []
    if int(os.environ["RANK"]) == 0:
        print("generating answers...")

    results = []
    for line in tqdm.tqdm(lines[start:end]):
        data = json.loads(line)
        question = data["question"]
        candidata_answer_1 = data["answer_1"]
        candidata_answer_2 = data["answer_2"]
        candidata_answer_3 = data["answer_3"]
        candidata_answer_4 = data["answer_4"]
        message_input = question + "\nNote that you need to summarize a more appropriate answer based on candidate answer 1 and the other three candidate answers: " + "Candidate answer 1: " + candidata_answer_1 + "Candidate answer 2: " + candidata_answer_2 + "Candidate answer 3: " + candidata_answer_3 + "Candidate answer 4: " + candidata_answer_4 + "Output your summarized answer for the provided image below:"
        # message_input = question.replace("Provide a one-sentence caption for the provided image.", "Provide a one-sentence caption for the provided image. Note that you need to summarize a more appropriate answer based on candidate answer 1 and the other three candidate answers: " + "Candidate answer 1: " + candidata_answer_1 + "Candidate answer 2: " + candidata_answer_2 + "Candidate answer 3: " + candidata_answer_3 + "Candidate answer 4: " + candidata_answer_4 + "Output your summarized one-sentence caption for the provided image below:")
        
        image = data["image_id"]
    
        image_path = os.path.join(args.image_path, image)
        image = Image.open(image_path).convert("RGB")
        conv = conv_templates[conv_mode].copy()
        image_tensor = process_images([image], image_processor, model.config)

        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        inp = message_input

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # print(prompt)
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
       
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=1.0,
                max_new_tokens=512,
                stopping_criteria=[stopping_criteria])
        ensemble_output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        

        if candidata_answer_1 != ensemble_output and ensemble_output != "No</s>" and ensemble_output != "Yes</s>" :
            # print("----------------------")
            # print("original output:" + candidata_answer_1)
            # print("ensemble output:" + ensemble_output)
            results.append({
                "chosen": ensemble_output,
                "reject": candidata_answer_1,
                "question": question,
                "image_id": data["image_id"]
            })
        elif candidata_answer_1 != candidata_answer_2:
            results.append({
                "chosen": candidata_answer_1,
                "reject": candidata_answer_2,
                "question": question,
                "image_id": data["image_id"]
            })
    device = f"cuda:{torch.cuda.current_device()}"
    # convert dictionary -> tensor for gather all results in all ranks
    part_tensor = convert_dict_to_tensor(results, device)
    shape_tensor = torch.tensor(part_tensor.shape, device=device)
    shape_list = [shape_tensor.clone() for _ in range(int(os.environ["WORLD_SIZE"]))]
    torch.distributed.all_gather(shape_list, shape_tensor)

    # gather tensor
    max_shape = max(shape_list)
    part_tensor_pad = torch.zeros(max_shape).to(device)
    part_tensor_pad[:part_tensor.shape[0]] = part_tensor
    tensor_list = [part_tensor_pad.clone() for _ in range(int(os.environ["WORLD_SIZE"]))]
    torch.distributed.all_gather(tensor_list, part_tensor_pad)

    if int(os.environ["RANK"]) == 0:
        results_all_rank = []
        for tensor, shape in zip(tensor_list, shape_list):
            t = tensor.long()[:shape]
            _data = "".join([chr(t[i].item()) for i in range(t.shape[0])])
            _data = json.loads(_data)
            results_all_rank.extend(_data)
        # sort according to question_id
        # results_all_rank = sorted(results_all_rank, key=lambda x:x["question_id"])
        res_file = args.res_file
        save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, res_file), "w") as f:
            for res in results_all_rank:
                f.write(json.dumps(res)+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--res_file", type=str, default="generate.jsonl")
    parser.add_argument('--augmentations', nargs='+', type=str, default=[])
    parser.add_argument("--noise_step", default=500, type=int)
    parser.add_argument("--image_file_list", default=None, type=str)
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()
    
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}, world {}): {}".format(
            int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), "env://"
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["LOCAL_RANK"]),
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    
    args = parser.parse_args()
    main(args)