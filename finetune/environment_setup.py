import subprocess

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}\n{stderr.decode()}")
    else:
        print(stdout.decode())
    return process.returncode

commands = [
    # "conda create -n hetarth_py10 python=3.10 -y",
    # "conda activate hetarth_py10",
    # "conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y",
    "conda install pytorch torchvision torchaudio -y",
    "conda install -c huggingface transformers -y",
    "pip install git+https://github.com/huggingface/peft.git",
    "pip install git+https://github.com/huggingface/transformers",
    "conda install -c huggingface -c conda-forge datasets -y",
    "conda install -c conda-forge accelerate -y",
    "conda install -c conda-forge huggingface_hub -y",
    "pip install bitsandbytes",
    "pip install wandb",
    "pip install scikit-learn"
]

for command in commands:
    if run_command(command) != 0:
        break
