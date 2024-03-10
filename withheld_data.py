import json
import random
#from tqdm import tqdm

# Load the data from the json file
with open('/u/bzd2/data/ftdata-new.json', 'r') as f:
    data = json.load(f)

# Calculate 10% of the total number of samples
num_samples = len(data)
num_samples_to_pick = int(num_samples * 0.1)

# Randomly pick 10% of the samples
assert type(data) is list
random.shuffle(data)
withheld_data = data[:num_samples_to_pick]
training_data = data[num_samples_to_pick:]

# Save the remaining data back to the original file in batches
with open('/u/bzd2/data/train_ftdata-new.json', 'w') as writefile:
    writefile.write(json.dumps(training_data))

with open('/u/bzd2/data/withheld_ftdata-new.json', 'w') as writefile:
    writefile.write(json.dumps(withheld_data))


