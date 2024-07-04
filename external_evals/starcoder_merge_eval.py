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
    parser.add_argument("--model_path", type=str, default="//projects/bbvz/choprahetarth/new_experiments/experiment_1/final_checkpoint_starcoderbase-1b_lr_0.0001_bs_64_ms_54_dp_/u/choprahetarth/all_files/data/train_ftdata-new-small.json")
    parser.add_argument("--save", type=str, default="/projects/bbvz/choprahetarth/mergekit_evals")
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
    args = get_args()
    model_path = "//projects/bbvz/choprahetarth/merged_models/gradient_merger"
    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    print("Dataloader created successfully.")
    
    results = []
    for batch in tqdm(dataloader, total=1646//args.batch_size):
        inputs = ["The ansible code for the following task is - "+row for row in batch['input']]
        responses = pipe(inputs)
        starcoder_response = [x[0]['generated_text'].split("Answer:", 1)[1] if "Answer:" in x[0]['generated_text'] else x[0]['generated_text'] for x in responses]
        codebertscore_precision =  [compute_similarity(row, starcoder_response[i])[0] for i, row in enumerate(batch['output'])]
        codebertscore_recall =  [compute_similarity(row, starcoder_response[i])[1] for i, row in enumerate(batch['output'])]
        codebertscore_f1 =  [compute_similarity(row, starcoder_response[i])[2] for i, row in enumerate(batch['output'])]
        sentence_bluescore =  [compute_bleu(row, starcoder_response[i]) for i, row in enumerate(batch['output'])]
        rouge1_precision =  [compute_rouge_scores(row, starcoder_response[i])['rouge1'].precision for i, row in enumerate(batch['output'])]
        rouge1_recall =  [compute_rouge_scores(row, starcoder_response[i])['rouge1'].recall for i, row in enumerate(batch['output'])]
        rouge1_fmeasure =  [compute_rouge_scores(row, starcoder_response[i])['rouge1'].fmeasure for i, row in enumerate(batch['output'])]
        rouge2_precision =  [compute_rouge_scores(row, starcoder_response[i])['rouge2'].precision for i, row in enumerate(batch['output'])]
        rouge2_recall =  [compute_rouge_scores(row, starcoder_response[i])['rouge2'].recall for i, row in enumerate(batch['output'])]
        rouge2_fmeasure =  [compute_rouge_scores(row, starcoder_response[i])['rouge2'].fmeasure for i, row in enumerate(batch['output'])]
        rougeL_precision =  [compute_rouge_scores(row, starcoder_response[i])['rougeL'].precision for i, row in enumerate(batch['output'])]
        rougeL_recall =  [compute_rouge_scores(row, starcoder_response[i])['rougeL'].recall for i, row in enumerate(batch['output'])]
        rougeL_fmeasure =  [compute_rouge_scores(row, starcoder_response[i])['rougeL'].fmeasure for i, row in enumerate(batch['output'])]
        crystal_bluescore =  [compute_crystal_bleu(row, starcoder_response[i]) for i, row in enumerate(batch['output'])]
        for i in range(len(starcoder_response)):
            results.append([starcoder_response[i],
                            codebertscore_precision[i],
                            codebertscore_recall[i],
                            codebertscore_f1[i],
                            sentence_bluescore[i],
                            crystal_bluescore[i],
                            rouge1_precision[i],
                            rouge1_recall[i],
                            rouge1_fmeasure[i],
                            rouge2_precision[i],
                            rouge2_recall[i],
                            rouge2_fmeasure[i],
                            rougeL_precision[i],
                            rougeL_recall[i],
                            rougeL_fmeasure[i]])

    # Convert the results to a DataFrame
    df = pd.DataFrame(results, columns=['starcoder_response', 
                                        'codebertscore_precision', 
                                        'codebertscore_recall', 
                                        'codebertscore_f1', 
                                        'sentence_bluescore', 
                                        'crystal_bluescore',
                                        'rouge1_precision', 
                                        'rouge1_recall', 
                                        'rouge1_fmeasure', 
                                        'rouge2_precision', 
                                        'rouge2_recall', 
                                        'rouge2_fmeasure', 
                                        'rougeL_precision', 
                                        'rougeL_recall', 
                                        'rougeL_fmeasure'])

    print(df.head)
    # Save the DataFrame to a csv file
    df.to_csv(f'{args.save}-merge_comparison.csv', index=True)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken for the script to run: {end_time - start_time} seconds")