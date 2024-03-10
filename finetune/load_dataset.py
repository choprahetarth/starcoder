from datasets import load_dataset

def main():
    # Load the dataset
    dataset = load_dataset('json', data_files='/u/bzd2/data/train_ftdata-new.json')

    # Print the total number of examples
    print(f"Total number of examples: {len(dataset)}")

if __name__ == "__main__":
    main()