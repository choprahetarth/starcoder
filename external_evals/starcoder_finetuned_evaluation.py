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
from vllm import LLM, SamplingParams

# from sacrebleu.metrics import BLEU, CHRF, TER
# ask dave to give the code for BLEU that he used, plus ansible aware
sampling_params = SamplingParams(temperature=0,  top_p=0.95)
def compute_rouge_scores(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

scorer = BERTScorer(lang="python")
def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--base_model_name_or_path", type=str, default="bigcode/starcoderbase-1b")
    # parser.add_argument("--peft_model_path", type=str, default="//projects/bbvz/choprahetarth/new_experiments/experiment_1/final_checkpoint_starcoderbase-1b_lr_0.0001_bs_64_ms_54_dp_/u/choprahetarth/all_files/data/train_ftdata-new-small.json")
    parser.add_argument("--model_path", type=str, default="/projects/bbvz/choprahetarth/experiment_1/fused_model")
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
    print("starting...")
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     args.base_model_name_or_path,
    #     return_dict=True,
    #     torch_dtype=torch.float16, # for starcoder7b/starcoder float 16, and others float32
    #     device_map='auto',
    # )
    
    # print("Loading PEFT model...")
    # model = PeftModel.from_pretrained(base_model, args.peft_model_path, device_map='auto')
    # print("Merging and unloading model...")
    # model = model.merge_and_unload()
    # print("Loading tokenizer...")
    # tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    
    # print("Saving pretrained model and tokenizer...")
    # model.save_pretrained(f"{args.save}-merged")
    # tokenizer.save_pretrained(f"{args.save}-merged")
    # print("Loading model for causal language modeling...")

    start_time = time.time()
    # model = AutoModelForCausalLM.from_pretrained(f"{args.save}-merged", device_map="auto", torch_dtype=torch.float16) #for starcoder7b 
    model = LLM(f"{args.model_path}",tensor_parallel_size=4, gpu_memory_utilization=0.8)
    print("Time taken to load model: ", time.time() - start_time)
    # print(f"Model saved to {args.peft_model_path}-merged")

    # pipe = pipeline("text-generation", model=model , tokenizer=tokenizer, max_length=512, device_map='auto')
    print("Loading dataset...")
    dataset = load_dataset('json', data_files='/u/choprahetarth/all_files/data/withheld_ftdata-new.json', streaming=True)
    dataset = dataset['train'].shuffle(seed=42).take(1646)
    print("Dataset loaded and shuffled successfully.")
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    results = []
    for batch in tqdm(dataloader, total=1646//args.batch_size):
        inputs = ["The ansible code for the following task is - "+row for row in batch['input']]
    # responses = pipe(inputs)
    responses = model.generate(inputs)
    # for output in responses:
    #     output_text = output.outputs[0].text
        
    for batch in tqdm(dataloader, total=1646//args.batch_size):
        responses = [x.outputs[0].text for x in responses]
        starcoder_response = [x.split("Answer:", 1)[1] if "Answer:" in x else x for x in responses]
        similarity_scores = [compute_similarity(row, starcoder_response[i]) for i, row in enumerate(batch['output'])]
        codebertscore_precision =  [score[0] for score in similarity_scores]
        codebertscore_recall =  [score[1] for score in similarity_scores]
        codebertscore_f1 =  [score[2] for score in similarity_scores]
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
    avg_row = ['0'] + df.iloc[:, 1:].mean(axis=0).tolist()
    df.loc[len(df)] = avg_row
    # df = df.append(avg_row, ignore_index=True)
    # Save the DataFrame to a csv file
    df.to_csv(f'{args.model_path}/finetuned_starcoder_comparison.csv', index=True, escapechar='\\')

if __name__ == "__main__" :
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken for the script to run: {end_time - start_time} seconds")



