import json

# Load the data from the json file
with open('/u/bzd2/data/train_ftdata-new.json', 'r') as f:
    training_data = json.load(f)

with open('/u/bzd2/data/withheld_ftdata-new.json', 'r') as f:
    withheld_data = json.load(f)

print(len(training_data))
print(len(withheld_data))

training_data = list(filter(lambda entry: int(entry["download_count"]) >= 500, training_data))
withheld_data = list(filter(lambda entry: int(entry["download_count"]) >= 500, withheld_data))

print(len(training_data))
print(len(withheld_data))


#dcs = {}
#for d in training_data:
#    dc = d["download_count"]
#    if dc not in dcs:
#        dcs[dc] = 1
#    else:
#        dcs[dc] += 1

#for i in sorted(map(int, list(dcs))):
#    print(i, dcs[str(i)])

# Save the remaining data back to the original file in batches
with open('/u/bzd2/data/train_ftdata-new-small.json', 'w') as writefile:
    writefile.write(json.dumps(training_data))

with open('/u/bzd2/data/withheld_ftdata-new-small.json', 'w') as writefile:
    writefile.write(json.dumps(withheld_data))

