import json
import pandas as pd
import os
from openai import OpenAI
from transformers import AutoTokenizer, pipeline
from peft import PeftModel
import torch
import code_bert_score
from nltk.translate.bleu_score import sentence_bleu
# Import tqdm for progress bar
from tqdm import tqdm

def compute_similarity(code1, code2):
    _, _, f1_score, _ = code_bert_score.score(cands=[code1], refs=[code2], lang='python')
    return f1_score.item()

def compute_bleu(reference, candidate):
    return sentence_bleu([reference], candidate)

def openai_main():
    print("Starting main function")
    # Load the json file
    with open('/u/bzd2/data/withheld_ftdata-new.json') as json_file:
        data = json.load(json_file)

    print("Loaded JSON file")
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    print("Converted data to DataFrame")
    # Shuffle the DataFrame with a fixed seed value for reproducibility
    df = df.sample(frac=1, random_state=42)
    print("Shuffled DataFrame")
    # Take the first 1646 samples
    df = df.iloc[:1646]
    print("Took first 1646 samples")

    # Initialize OpenAI client
    client = OpenAI()
    print("Initialized OpenAI client")

    # Create a new column 'gpt response' to store the generated responses
    df['gpt response'] = ''
    df['codebertscore'] = 0.0
    df['bleuscore'] = 0.0
    cost = 0 
    print("Initialized new columns and cost variable")

    # Iterate over the 'input' column of the DataFrame
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        print(f"Processing row {index}")
        # Add the input to the context of openai generation
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a code completion co-pilot that takes in an ansible task's name and your objective is to just print the code for completing the task. DON'T PRINT THE ENTIRE PLAYBOOK, JUST THE TASK. Just print the code output and no remarks whatsoever. DON'T PRINT THE ANSIBLE NAME FILE, JUST COMPLETE THE CODE."},
                {"role": "user", "content":  row['input']}
            ],
            temperature=1
        )
        print("Generated completion")

        # Store the generated answer in the 'gpt response' column
        df.at[index, 'gpt response'] = completion.choices[0].message.content
        df.at[index, 'codebertscore'] = compute_similarity(row['output'], completion.choices[0].message.content)
        df.at[index, 'bleuscore'] = compute_bleu(row['output'], completion.choices[0].message.content)
        cost+=int(completion.usage.total_tokens)*(0.06/1000)
        print(f"Updated DataFrame and cost for row {index}")
        print(cost)
        # print(f"Tokens used for query {index}: {completion['usage']['total_tokens']}")

    # Save the DataFrame to a csv file
    df.to_csv('openai_comparison_gpt-3.5-turbo-new.csv', index=False)
    print("Saved DataFrame to CSV file")
    df.head()

if __name__ == "__main__":
    print("Starting script")
    openai_main()
