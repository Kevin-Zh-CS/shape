import json


input_files = ["textvqa_answer_file_8k_base.jsonl", "textvqa_answer_file_8k_contrast.jsonl", "textvqa_answer_file_8k_diffusion_step500.jsonl", "textvqa_answer_file_8k_flip.jsonl"] 
output_file = "textvqa_answer_file_8k_merged.jsonl" 

merged_data = {}

for idx, filename in enumerate(input_files):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())  
            question = data["question"]
            image_id = data["image_id"]
            answer = data["answer"]

            key = (question, image_id)

            if key not in merged_data:
                merged_data[key] = {
                    "question": question,
                    "image_id": image_id + ".jpg"
                }

            merged_data[key][f"answer_{idx+1}"] = answer

with open(output_file, 'w', encoding='utf-8') as f_out:
    for item in merged_data.values():
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"File has been saved at: {output_file}")