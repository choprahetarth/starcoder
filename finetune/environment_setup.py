#!/usr/bin/env python3

print("Starting script...")

import subprocess

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(f"Output: {stdout.decode()}")
    if process.returncode != 0:
        print(f"Error executing command: {command}\n{stderr.decode()}")
    return process.returncode

commands = [
    "pip install torch torchvision torchaudio",
    "pip install transformers",
    "echo 'y' | pip uninstall peft",
    "pip install git+https://github.com/huggingface/peft.git",
    "pip install git+https://github.com/huggingface/transformers",
    "pip install datasets",
    "pip install accelerate",
    "pip install huggingface_hub",
    "pip install bitsandbytes",
    "pip install wandb",
    "pip install scikit-learn",
    "pip install code_bert_score",
    "pip install nltk",
]

for command in commands:
    if run_command(command) != 0:
        break