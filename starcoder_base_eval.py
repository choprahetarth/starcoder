from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
import json
import pandas as pd
import os
import code_bert_score
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from datasets import load_dataset

def compute_similarity(code1, code2):
    _, _, f1_score, _ = code_bert_score.score(cands=[code1], refs=[code2], lang='python')
    return f1_score.item()

def compute_bleu(reference, candidate):
    return sentence_bleu([reference], candidate)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-3b")
    model = AutoModelForCausalLM.from_pretrained("bigcode/starcoderbase-3b")
    model = model.to(device)

    pipe = pipeline("text-generation", model=model , tokenizer=tokenizer, max_length=512, device='cuda')
    dataset = load_dataset('json', data_files='/u/bzd2/data/withheld_ftdata-new.json', streaming=True)
    dataset = dataset['train'].shuffle(seed=42).take(1646)
    results = []
    for row in tqdm(dataset, total=1646):
        response = pipe("The ansible code for the following task is - "+row['input'])
        row['starcoder-response'] = response[0]['generated_text']
        # Compute the scores
        row['codebertscore'] = compute_similarity(row['output'], row['starcoder-response'])
        row['bleuscore'] = compute_bleu(row['output'], row['starcoder-response'])
        results.append(row)

    # Convert the results to a DataFrame
    df = pd.DataFrame(results)
    # Save the DataFrame to a csv file
    df.to_csv("starcoder_baseline_eval_3b_16fp_with_prompt.csv", index=True)

if __name__ == "__main__" :
    main()