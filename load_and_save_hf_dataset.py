import os
from datasets import load_dataset

def save_hf_dataset_to_csv(dataset_name, config_name=None,
                           output_dir="./datasets", output_prefix=None):
    """
    Loads a Hugging Face dataset (optionally with a config) and saves each split as CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    if config_name:
        dataset = load_dataset(dataset_name, config_name)
    else:
        dataset = load_dataset(dataset_name)

    # Default prefix is dataset_name + config if provided
    if output_prefix is None:
        prefix = dataset_name.replace("/", "_").replace("-", "_")
        if config_name:
            prefix += f"_{config_name}"
        output_prefix = prefix

    # Save each split
    for split_name, split in dataset.items():
        filename = os.path.join(output_dir, f"{output_prefix}_{split_name}.csv")
        print(f"Saving {split_name} to {filename} ...")
        split.to_csv(filename)

    print("Done!")

if __name__ == "__main__":
    # save_hf_dataset_to_csv("HuggingFaceH4/MATH-500")
    # save_hf_dataset_to_csv("HuggingFaceH4/aime_2024")
    # save_hf_dataset_to_csv("openai/gsm8k", config_name="main")
    from datasets import get_dataset_split_names
    print(get_dataset_split_names("HuggingFaceH4/MATH-500"))
    print(get_dataset_split_names("HuggingFaceH4/aime_2024"))
