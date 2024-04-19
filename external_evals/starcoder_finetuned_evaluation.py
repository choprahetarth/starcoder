from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
from torch.utils.data import DataLoader
import pandas as pd
import argparse
# import code_bert_score
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from datasets import load_dataset
from code_bert_score import BERTScorer
from rouge_score import rouge_scorer
# from sacrebleu.metrics import BLEU, CHRF, TER
# ask dave to give the code for BLEU that he used, plus ansible aware

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
    precision, recall, f1_score, f3_score = scorer.score(cands=[code1], refs=[code2], lang='python')
    return (precision.item(),recall.item(),f1_score.item(), f3_score.item())

def compute_bleu(reference, candidate):
    return sentence_bleu([reference], candidate)

def main():
    args = get_args()
    print("starting...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float32,
        device_map='auto'
    )
    
    print("Loading PEFT model...")
    model = PeftModel.from_pretrained(base_model, args.peft_model_path, device_map='auto')
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

    pipe = pipeline("text-generation", model=model , tokenizer=tokenizer, max_length=512, device_map='auto')
    print("Loading dataset...")
    dataset = load_dataset('json', data_files='/u/choprahetarth/all_files/data/withheld_ftdata-new.json', streaming=True)
    dataset = dataset['train'].shuffle(seed=42).take(1646)
    print("Dataset loaded and shuffled successfully.")
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    results = []
    for batch in tqdm(dataloader, total=1646//args.batch_size):
        inputs = ["The ansible code for the following task is - "+row for row in batch['input']]
        responses = pipe(inputs)
        starcoder_response = [x[0]['generated_text'].split("Answer:", 1)[1] if "Answer:" in x[0]['generated_text'] else x[0]['generated_text'] for x in responses]
        codebertscore_precision =  [compute_similarity(row, starcoder_response[i])[0] for i, row in enumerate(batch['output'])]
        codebertscore_recall =  [compute_similarity(row, starcoder_response[i])[1] for i, row in enumerate(batch['output'])]
        codebertscore_f1 =  [compute_similarity(row, starcoder_response[i])[2] for i, row in enumerate(batch['output'])]
        codebertscore_f3 =  [compute_similarity(row, starcoder_response[i])[3] for i, row in enumerate(batch['output'])]
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
        for i in range(len(starcoder_response)):
            results.append([starcoder_response[i],
                            codebertscore_precision[i],
                            codebertscore_recall[i],
                            codebertscore_f1[i],
                            codebertscore_f3[i],
                            sentence_bluescore[i],
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
                                        'codebertscore_f3', 
                                        'sentence_bluescore', 
                                        'rouge1_precision', 
                                        'rouge1_recall', 
                                        'rouge1_fmeasure', 
                                        'rouge2_precision', 
                                        'rouge2_recall', 
                                        'rouge2_fmeasure', 
                                        'rougeL_precision', 
                                        'rougeL_recall', 
                                        'rougeL_fmeasure'])
    # Save the DataFrame to a csv file
    df.to_csv(f'{args.save}-finetuned_starcoder_comparison.csv', index=True)

if __name__ == "__main__" :
    main()


