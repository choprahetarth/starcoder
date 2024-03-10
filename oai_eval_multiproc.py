import json
import pandas as pd
import os
from openai import OpenAI
from transformers import AutoTokenizer, pipeline
from peft import PeftModel
import torch
import code_bert_score
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from multiprocessing import Pool

client = OpenAI()
print("Initialized OpenAI client")

def compute_similarity(code1, code2):
    _, _, f1_score, _ = code_bert_score.score(cands=[code1], refs=[code2], lang='python')
    return f1_score.item()

def compute_bleu(reference, candidate):
    return sentence_bleu([reference], candidate)

def process_row(row):
    index, row = row
    print(f"Processing row {index}")
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a code completion co-pilot that takes in an ansible task's name and your objective is to just print the code for completing the task. DON'T PRINT THE ENTIRE PLAYBOOK, JUST THE TASK. Just print the code output and no remarks whatsoever. DON'T PRINT THE ANSIBLE NAME FILE, JUST COMPLETE THE CODE."},
            {"role": "user", "content":  row['input']}
        ],
        temperature=1
    )
    print("Generated completion")
    row['gpt response'] = completion.choices[0].message.content
    row['codebertscore'] = compute_similarity(row['output'], completion.choices[0].message.content)
    row['bleuscore'] = compute_bleu(row['output'], completion.choices[0].message.content)
    cost = int(completion.usage.total_tokens)*(0.06/1000)
    print(f"Updated DataFrame and cost for row {index}")
    print(cost)
    return row

def openai_main():
    print("Starting main function")
    with open('/u/bzd2/data/withheld_ftdata-new.json') as json_file:
        data = json.load(json_file)

    print("Loaded JSON file")
    df = pd.DataFrame(data)
    print("Converted data to DataFrame")
    df = df.sample(frac=1, random_state=42)
    print("Shuffled DataFrame")
    df = df.iloc[:1646]
    print("Took first 1646 samples")

    df['gpt response'] = ''
    df['codebertscore'] = 0.0
    df['bleuscore'] = 0.0
    cost = 0 
    print("Initialized new columns and cost variable")

    with Pool(processes=4) as pool:  # Use 4 processes
        for i, result in enumerate(tqdm(pool.imap(process_row, df.iterrows()), total=len(df))):
            df.iloc[i] = result

    df.to_csv('openai_comparison_gpt-4-new-multiproc.csv', index=False)
    print("Saved DataFrame to CSV file")
    df.head()

if __name__ == "__main__":
    print("Starting script")
    openai_main()