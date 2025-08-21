from datasets import load_dataset

def download_dataset():
    """
    Downloads the KazMIRIAD dataset from Hugging Face.
    """
    dataset = load_dataset("miriad/miriad-4.4M", split="train")

    # save this dataset to disk as csv preserving the labels
    dataset.to_csv("./MIRIAD4_4_dataset.csv", index=True)

    print("Dataset downloaded and saved as MIRIAD4_4_dataset.csv")

download_dataset()