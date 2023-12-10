from datasets import load_dataset

# Load the dataset
dataset = load_dataset('json', data_files='/home/bzd2/ansible-scraping/data/ftdata.json')
print(dataset)
# Print the total number of samples in the dataset
print(dataset['train'].num_rows)