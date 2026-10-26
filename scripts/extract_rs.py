import argparse
import os

import pandas as pd
import torch
from src.activations import ActivationCache
from src.global_settings import settings
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = settings.lm_weights_path
TEXT_DATASET_PATH = settings.text_dataset_path
SAVE_DIR = settings.rs_data_dir

DEFAULT_BATCH_SIZE = settings.prev_bs  # 64
DEFAULT_MAX_LENGTH = settings.prev_seq_len  # 1024
DEFAULT_DTYPE = settings.prev_dtype  # float16

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_and_save_activations_batch(
    text_dataset: pd.DataFrame,
    activation_cache: ActivationCache,
    tokenizer: AutoTokenizer,
    target_layers: list[int],
    save_dir: str,
    batch_size: int,
    max_length: int,
    dtype: torch.dtype,
    device: torch.device = DEFAULT_DEVICE,
):
    """
    Extracts Residual Stream (also called 'Activations') vectors from language model (i.e. classifier) to be analyzed by SAE,
    and saves them as 'batch_{i}.pt' files to 'save_dir/layer_{k}/' directories, with a JSON metadata file, which maps
    each batch file to the original text sample indices in 'text_dataset'.

    Note that text_dataset is fully being processed, without any sampling.
    """

    assert "text" in text_dataset.columns, "text_dataset must contain column `text`."
    assert device.type == "cuda", "CUDA not available, check device or disable this assertion."

    for layer_idx in target_layers:
        layer_dir = os.path.join(save_dir, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)

    texts = text_dataset["text"].tolist()
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(total_batches), desc="Extracting activations by batches..."):
        batch_indices = text_dataset.index[i * batch_size : (i + 1) * batch_size].tolist()
        batch_texts = texts[i * batch_size : (i + 1) * batch_size]

        if not batch_texts:
            continue

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)

        with activation_cache.capture(inputs, layer_indices=target_layers) as activations:
            for layer_idx, tensor in activations.items():
                # reshaped_tensor = tensor.view(-1, tensor.shape[-1])  # (batch_size * seq_len, hidden_size)

                attention_mask = inputs["attention_mask"]  # 2025-09-30: attention_mask added

                data_ts = {
                    "indices": torch.tensor(batch_indices, dtype=torch.long),
                    "activations": tensor.to(dtype).cpu(),
                    "attention_mask": attention_mask.cpu(),
                }

                layer_dir = os.path.join(save_dir, f"layer_{layer_idx}")
                save_path = os.path.join(layer_dir, f"batch_{i}.pt")
                torch.save(data_ts, save_path)

    print(f"Extraction process completed. Tensor files have been saved to: {os.path.abspath(save_dir)}")


def main(args):
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(DEFAULT_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    activation_cache = ActivationCache(model)

    text_dataset = pd.read_csv(args.text_dataset_path)
    print(f"Dataset loaded with {len(text_dataset):,} samples.")

    if isinstance(args.dtype, str):
        args.dtype = getattr(torch, args.dtype)

    assert isinstance(args.dtype, torch.dtype), f"Invalid type of `dtype`: {type(args.dtype)} ({args.dtype})"

    extract_and_save_activations_batch(
        text_dataset=text_dataset,
        activation_cache=activation_cache,
        tokenizer=tokenizer,
        target_layers=args.target_layers,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
        device=DEFAULT_DEVICE,
    )


if __name__ == "__main__":
    """
    Usage Example:
        python -m scripts.extract_rs --target_layers 1 6 15

    Output Example:
        --- Running with Arguments ---
        target_layers  : [1, 6, 15]
        model_path     : ...
        text_dataset_path: .../data/text_dataset/textset_balanced.csv
        save_dir       : .../data/post_mlp_residual_stream
        batch_size     : 64
        max_length     : 1024
        dtype          : torch.float16
        ------------------------------
    """

    parser = argparse.ArgumentParser(description="Extract and save model activations for SAE training.")

    parser.add_argument(
        "--target_layers",
        type=int,
        nargs="+",
        required=True,
        help="List of layer indices to extract activations from. (e.g., --target_layers 1 11 20)",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to the pre-trained model.",
    )
    parser.add_argument(
        "--text_dataset_path",
        type=str,
        default=TEXT_DATASET_PATH,
        help="Path to the CSV file containing the text dataset.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=SAVE_DIR,
        help="Directory to save the extracted activation files.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing the dataset. (default: 16)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Maximum sequence length for tokenization. (default: {DEFAULT_MAX_LENGTH})",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        help="Data type for storing activations (e.g., float16, float32). (default: float16)",
    )

    args = parser.parse_args()
    print("--- Running with Arguments ---")
    for k, v in vars(args).items():
        print(f"{k:<15}: {v}")
    print("-" * 30)

    main(args)
