import requests
import json
import re
import pandas as pd
import os
from transformers import AutoTokenizer, pipeline
from peft import PeftModel
import torch
import code_bert_score
from nltk.translate.bleu_score import sentence_bleu
from multiprocessing import Pool
from tqdm import tqdm

def compute_similarity(code1, code2):
    _, _, f1_score, _ = code_bert_score.score(cands=[code1], refs=[code2], lang='python')
    return f1_score.item()

def compute_bleu(reference, candidate):
    return sentence_bleu([reference], candidate)

def extract_code_block(response_text):
    pattern = r"```yaml(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

def process_row(row):
    index, row = row
            # Add the input to the context of openai generation
    url = "https://www.uiuc.chat/api/chat"
    content_for_sending = str(row['input'])+"\n"
    url_context = f"https://flask-production-751b.up.railway.app/getTopContexts?course_name=cropwizard&search_query={content_for_sending}%0A&token_limit=8192"
    payload_context = {}
    headers_context = {
    'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.uiuc.chat/',
    'sec-ch-ua-mobile': '?1',
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
    'sec-ch-ua-platform': '"Android"'
    }

    context_response = requests.request("GET", url_context, headers=headers_context, data=payload_context)

    payload = json.dumps({
    "model": {
        "id": "gpt-4-from-canada-east",
        "name": "GPT-4",
        "maxLength": 24000,
        "tokenLimit": 8192
    },
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": content_for_sending
            }
        ],
        "contexts":eval(context_response.text)
        }
    ],
    "key": "+6OFuMjY4Rda4InTnvJBCuWxsZ2gEt3h35YHyJRWRJxKEsHNeze1CEIQ1hredHs5.8tCGp/TlmLvcFCPg",
    "prompt": "You are a code completion co-pilot that takes in an ansible task's name and your objective is to just print the code for completing the task. DON'T PRINT THE ENTIRE PLAYBOOK, JUST THE TASK. Just print the code output and no remarks whatsoever. DON'T PRINT THE ANSIBLE NAME FILE, JUST COMPLETE THE CODE.",
    "temperature": 1,
    "course_name": "ansible2",
    "stream": True
    })
    headers = {
    'authority': 'www.uiuc.chat',
    'accept': '*/*',
    'accept-language': 'en-IN,en-US;q=0.9,en-GB;q=0.8,en;q=0.7',
    'content-type': 'application/json',
    'cookie': '__client_uat=1700704543; __clerk_db_jwt=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJkZXYiOiJkdmJfMlVmbm5lYnJ0OEdxNjN4WHJzNTB2TWNLM3cwIiwiaWQiOiJjbGllbnRfMlV6Y0ZPUncyem0xTUF3cUNhZUxoclFpelJoIiwicm90YXRpbmdfdG9rZW4iOiJuNTNjOWlzbGd2MTg2a3QxcHo3ODFibjdudTcwMTcxdG1oNjRrMTZ0In0.SzuX2gx1Du3JJGafrW9ojIWuDizWTlDD-FaBJGNs3RTKiVD-h5EkTCyLTYM1gbZbBuBHz91lruoaF16hmfSbnCGN75w3ua9iQjaahUBwIMUHg1zINoe5xVVlOfKV5IYSsrXK-IKrWYFI0nn2pSZt03xrdFpKaNNScKx7JNZdi_Hu4zxLZn-2ELdFVZeG-twDpBVqU7VOiMu35D0sK4wfjtuhQwfkZNslOFrf3GmwsgjzPgZl4Er4M424yFET5E4eeYWQJ5zY8dWobNx4nlXj0TtDOSWPB6ldEBS7Aiqq8jEXLcg7JdPVNvY8FUxa_TtkOJe-IEQp6hEpOkax_RjtEw; __session=eyJhbGciOiJSUzI1NiIsImtpZCI6Imluc18yUExpNHBGZlBVN01oT3IxbmxlVjhTSzBoTmUiLCJ0eXAiOiJKV1QifQ.eyJhenAiOiJodHRwczovL3d3dy51aXVjLmNoYXQiLCJleHAiOjE3MDI1ODU4NzYsImlhdCI6MTcwMjU4NTgxNiwiaXNzIjoiaHR0cHM6Ly9mb25kLWphdmVsaW4tNy5jbGVyay5hY2NvdW50cy5kZXYiLCJuYmYiOjE3MDI1ODU4MDYsInNpZCI6InNlc3NfMllZWmJRMGdiRmNDNkNvd3dZR2gza21zM3k3Iiwic3ViIjoidXNlcl8yVXpjR0s1MWtOVU1LSmNOaWNOYlAyU3N5clkifQ.N3djtWXvxqiQZUf_MscqkKqPYkt68HbhbV6I8tbh_0g5zKtYrKJD01K5TJmLmkqtxHsCyrr8arzd3lJme49l_A0dLRBcfOu27uRSm1nBj3RAbqY8AGxknUVDYTM7v3dWE57TvpeIOCKdMdN0E9aIHzu5uo_naRpkNBVP0nRgGXwYwTE9FGt4NGUv62y2zgSz1-n4M3ezySaQfdu4_TEkQAwq-vqhasGKDaPbtLZHPbrCGAuoqMS7Cn-gSNCINPKEdP0NlI8YxddKC8lNqa9QF7Vab1Py-sNSrX8HU9Rph_-LvRL0wgsTG8zL6EBuRDJeBKphZU7ePd-OImiA6sPeLA',
    'origin': 'https://www.uiuc.chat',
    'referer': 'https://www.uiuc.chat/ansible2/chat',
    'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    'sec-ch-ua-mobile': '?1',
    'sec-ch-ua-platform': '"Android"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    code_output = extract_code_block(response.text)
    row['uiuc_chat_response'] = code_output
    print(code_output)
    if code_output is not None:
        row['codebertscore'] = compute_similarity(row['output'], code_output)
        row['bleuscore'] = compute_bleu(row['output'], code_output)
    else:
        row['codebertscore'] = 0.0  # or some other default value
        row['bleuscore'] = 0.0  
    return row

def openai_main():
    with open('/u/bzd2/data/withheld_ftdata-new-small.json') as json_file:
        data = json.load(json_file)

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42)
    df = df.iloc[:1646]

    df['uiuc_chat_response'] = ''
    df['codebertscore'] = 0.0
    df['bleuscore'] = 0.0

    with Pool(processes=4) as pool:  # Use 4 processes
        results = []
        for i, row in enumerate(tqdm(df.iterrows(), total=len(df))):
            result = pool.apply_async(process_row, args=(row,))
            results.append(result)
        for i, result in enumerate(results):
            df.iloc[i] = result.get()

    df.to_csv('uiuc_chat_comparison-new-small.csv', index=False)
    df.head()

if __name__ == "__main__":
    openai_main()



