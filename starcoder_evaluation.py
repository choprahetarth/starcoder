from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
import json
import pandas as pd
import os
import argparse
import code_bert_score
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from datasets import load_dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default="bigcode/large-model")
    parser.add_argument("--peft_model_path", type=str, default="/projects/bbvz/bzd2/checkpoints_experiment_1/checkpoint-500")
    parser.add_argument("--push_to_hub", action="store_true", default=True)
    parser.add_argument("--save", type=str, default="/projects/bbvz/bzd2/checkpoints_experiment_4/fused_model")

    return parser.parse_args()

def compute_similarity(code1, code2):
    _, _, f1_score, _ = code_bert_score.score(cands=[code1], refs=[code2], lang='python')
    return f1_score.item()

def compute_bleu(reference, candidate):
    return sentence_bleu([reference], candidate)

def main():
    args = get_args()
    print("starting...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float32
    )

    print("Loading PEFT model...")
    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    print("Merging and unloading model...")
    model = model.merge_and_unload()
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    print("Saving pretrained model and tokenizer...")
    model.save_pretrained(f"{args.save}-merged")
    tokenizer.save_pretrained(f"{args.save}-merged")
    print("Loading model for causal language modeling...")
    model = AutoModelForCausalLM.from_pretrained(f"{args.save}-merged", device_map="auto")
    print(f"Model saved to {args.peft_model_path}-merged")

    pipe = pipeline("text-generation", model=model , tokenizer=tokenizer, max_length=1024)

    # Load the json file using the datasets library in a streaming manner
    dataset = load_dataset('json', data_files='/u/bzd2/data/withheld_ftdata-new.json', streaming=True)

    # Randomly select 1646 samples with a fixed seed value for reproducibility
    dataset = dataset['train'].shuffle(seed=42).take(1646)

    # Create a new column 'starcoder response' to store the generated responses
    # Process the data one by one
    results = []
    for row in tqdm(dataset, total=1646):
        response = pipe(row['input'])
        row['starcoder-response'] = response[0]['generated_text']

        # Compute the scores
        row['codebertscore'] = compute_similarity(row['output'], row['starcoder-response'])
        row['bleuscore'] = compute_bleu(row['output'], row['starcoder-response'])
        results.append(row)

    # Convert the results to a DataFrame
    df = pd.DataFrame(results)

    # Save the DataFrame to a csv file
    df.to_csv(f'{args.save}-merged_starcoder_comparison.csv', index=True)

if __name__ == "__main__" :
    main()