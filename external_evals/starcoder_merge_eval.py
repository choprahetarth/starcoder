import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
import pandas as pd
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
    model_path = "//projects/bbvz/choprahetarth/merged_models/gradient_merger"
    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    print("Model loaded successfully.")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded successfully.")

    print("Creating pipeline...")
    pipe = pipeline("text-generation", model=model , tokenizer=tokenizer, max_length=512, device_map='auto')
    print("Pipeline created successfully.")

    print("Loading dataset...")
    dataset = load_dataset('json', data_files='/u/choprahetarth/all_files/data/withheld_ftdata-new.json', streaming=True)
    dataset = dataset['train'].shuffle(seed=42).take(1646)
    print("Dataset loaded and shuffled successfully.")

    print("Creating dataloader...")
    dataloader = DataLoader(dataset, batch_size=32)
    print("Dataloader created successfully.")
    results = []
    for batch in tqdm(dataloader, total=1646//32):
        inputs = ["The ansible code for the following task is - "+row for row in batch['input']]
        responses = pipe(inputs)
        starcoder_response = [x[0]['generated_text'].split("Answer:", 1)[1] if "Answer:" in x[0]['generated_text'] else x[0]['generated_text'] for x in responses]
        codebertscore =  [compute_similarity(row, starcoder_response[i]) for i, row in enumerate(batch['output'])]
        bluescore =  [compute_bleu(row, starcoder_response[i]) for i, row in enumerate(batch['output'])]
        for i in range(len(starcoder_response)):
            results.append([starcoder_response[i], codebertscore[i], bluescore[i]])

    # Convert the results to a DataFrame
    df = pd.DataFrame(results, columns=['starcoder_response', 'codebertscore', 'bluescore'])
    print(df.head)
    # Save the DataFrame to a csv file
    df.to_csv(f'merged1bs_ties_starcoder_comparison_2_gradient.csv', index=True)


if __name__ == "__main__":
    main()