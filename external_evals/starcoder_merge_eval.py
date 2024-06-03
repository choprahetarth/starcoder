import time
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from collections import Counter
from nltk.util import ngrams
from peft import PeftModel
from datasets import load_dataset
from rouge_score import rouge_scorer
from code_bert_score import BERTScorer
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from crystalbleu import corpus_bleu

def compute_rouge_scores(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

scorer = BERTScorer(lang="python")
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default="bigcode/starcoderbase-1b")
    parser.add_argument("--peft_model_path", type=str, default="//projects/bbvz/choprahetarth/new_experiments/experiment_1/final_checkpoint_starcoderbase-1b_lr_0.0001_bs_64_ms_54_dp_/u/choprahetarth/all_files/data/train_ftdata-new-small.json")
    parser.add_argument("--save", type=str, default="/projects/bbvz/choprahetarth/experiment_1/fused_model")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()

def compute_similarity(code1, code2):
    precision, recall, f1_score = scorer.score(cands=[code1], refs=[code2])
    return (precision.item(),recall.item(),f1_score.item())

def compute_bleu(reference, candidate):
    return sentence_bleu([reference], candidate)

def compute_crystal_bleu(reference, candidate):
    k = 500
    all_ngrams = []
    for n in range(1, 5):
        all_ngrams.extend(list(ngrams(reference.split(), n)))
    frequencies = Counter(all_ngrams)
    trivially_shared_ngrams = dict(frequencies.most_common(k))
    crystalBLEU_score = corpus_bleu([reference], [candidate], ignoring=trivially_shared_ngrams)
    return crystalBLEU_score

def main():
    model_path = "//projects/bbvz/choprahetarth/merged_models/gradient_merger"
    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    print("Model loaded successfully.")

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