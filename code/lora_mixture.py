import os
import json
import ast
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Constants
DATA_DIR = "./data/test_data/v/"
WEIGHTS_PATH = "./data/test_data/encoder_eucl_weights.json"
PEFT_MODEL_DIR = "../LLaMA-Factory/saves"
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
FILE_LIST = ["EXAMPLE_DATA.csv"]
OUTPUT_DIR = "./data/results_lora/"
LORA_ADAPTERS = [
    "AAD", "BUS", "ENG", "LAHS", "NRE", "SCI", "VM", "ALS",
    "elife", "cells", "news", "plos", "all"
]

# ==== Load Models ====
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map="cuda", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, f"{PEFT_MODEL_DIR}/all_2/lora/sft", adapter_name="all")

# Load additional adapters
for adapter in LORA_ADAPTERS:
    if adapter != "all":
        model.load_adapter(f"{PEFT_MODEL_DIR}/{adapter}/lora/sft", adapter_name=adapter)

# Load weights
with open(WEIGHTS_PATH, "r") as f:
    weights_data = json.load(f)

# ====Sci-LoRA Architecture====
def generate_output(file_path, filename):
    file = pd.read_csv(file_path)
    abstracts_raw = file['abstract'].tolist()
    summaries_raw = file['abstract_general'].tolist()

    outputs, labels, abstracts = [], [], []

    for i, (abs_str, sum_str) in enumerate(zip(abstracts_raw, summaries_raw)):
        abstract = ast.literal_eval(abs_str)[0]
        summary = ast.literal_eval(sum_str)[0]
        prompt = f"Generate another version of the provided text for general audiences: {abstract}"

        messages = [
            {"role": "system", "content": "You are an expert on technical to general audience text paraphrasing tasks."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Get top LoRA weights
        lora_weights = [float(v) for v in weights_data[filename][str(i)].values()]
        top_indices = sorted(range(len(lora_weights)), key=lambda j: lora_weights[j], reverse=True)[:12]
        selected_adapters = ["all"] + [LORA_ADAPTERS[j] for j in top_indices]
        selected_weights = [1.0] + [lora_weights[j] for j in top_indices]

        # Add weighted adapter
        fused_adapter_name = "fused_lora"
        model.add_weighted_adapter(
            adapters=selected_adapters,
            weights=selected_weights,
            adapter_name=fused_adapter_name,
            combination_type="linear"
        )
        model.set_adapter(fused_adapter_name)

        # Get embedding from fused LoRA
        with torch.no_grad():
            fused_emb = model.get_input_embeddings()(model_inputs.input_ids)

        model.delete_adapter(fused_adapter_name)

        # Add embedding from top individual adapter
        individual_adapter = selected_adapters[1]
        model.set_adapter(individual_adapter)
        with torch.no_grad():
            individual_emb = model.get_input_embeddings()(model_inputs.input_ids)

        # Combine embeddings (weighted average)
        combined_emb = (fused_emb + individual_emb) / 2

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(inputs_embeds=combined_emb, max_new_tokens=1024)

        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        outputs.append(response)
        labels.append(summary)
        abstracts.append(abstract)

    return pd.DataFrame({
        "Generated_Output": outputs,
        "Labels": labels,
        "Input": abstracts
    })

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in FILE_LIST:
        file_path = os.path.join(DATA_DIR, filename)
        df_result = generate_output(file_path, filename)
        output_path = os.path.join(OUTPUT_DIR, f"fusion_lora_{filename}")
        df_result.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
