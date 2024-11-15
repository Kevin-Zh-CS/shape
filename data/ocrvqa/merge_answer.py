# import json


# input_files = ["ocrvqa_answer_file_8k_base.jsonl", "ocrvqa_answer_file_8k_contrast.jsonl", "ocrvqa_answer_file_8k_diffusion_step_500.jsonl", "ocrvqa_answer_file_8k_flip.jsonl"] 
# output_file = "ocrvqa_answer_file_8k_merged.jsonl" 

# merged_data = {}

# for idx, filename in enumerate(input_files):
#     with open(filename, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line.strip())  
#             question = data["question"]
#             image_id = data["image_id"]
#             answer = data["answer"]

#             key = (question, image_id)

#             if key not in merged_data:
#                 merged_data[key] = {
#                     "question": question,
#                     "image_id": image_id + ".jpg"
#                 }

#             merged_data[key][f"answer_{idx+1}"] = answer

# with open(output_file, 'w', encoding='utf-8') as f_out:
#     for item in merged_data.values():
#         f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

# print(f"File has been saved at: {output_file}")

import json

def process_jsonl_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            item = json.loads(line)
            # Swap chosen and reject if chosen is "Yes</s>" or "No</s>"
            if item["chosen"] in ["Yes</s>", "No</s>"]:
                item["chosen"], item["reject"] = item["reject"], item["chosen"]
            # Write the processed item to the output file
            outfile.write(json.dumps(item) + '\n')

# 使用方法
input_file = 'ocrvqa_answer_file_8k_dpo.jsonl'   # 输入文件路径
output_file = 'ocrvqa_answer_file_8k_dpo_2.jsonl' # 输出文件路径
process_jsonl_file(input_file, output_file)
