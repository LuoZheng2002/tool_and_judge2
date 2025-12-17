from datasets import load_dataset
import json
def download_mmmlu_dataset(lang: str, output_file: str):
    print(f"Downloading MMLU intersection filtered dataset for language: {lang}")
    ds = load_dataset("willchow66/mmmlu-intersection-filtered", lang)
    train_split = ds['train']
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in train_split:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved dataset to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download MMLU intersection filtered dataset for a specified language.")
    parser.add_argument('--lang', type=str, required=True, help='Language code (e.g., zh, es, fr)')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path to save the dataset')
    args = parser.parse_args()
    
    download_mmmlu_dataset(args.lang, args.output_file)