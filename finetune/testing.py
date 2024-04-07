import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the cache directory
os.environ["HF_HOME"] = "/projects/bbvz/bzd2"
os.environ["TRANSFORMERS_CACHE"] = "/projects/bbvz/bzd2"

# Load the model and tokenizer
model_name = "distilgpt2"  # small model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the model is in evaluation mode and moved to CPU
model.eval()
model.to("cpu")

# Now you can use the model for inference
# For example:
input_text = "Hello, how are you?"
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_length=50)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)
