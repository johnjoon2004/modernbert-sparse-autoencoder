import pandas as pd
import torch
import torch.nn.functional as F
from src.global_settings import settings
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    model = AutoModelForSequenceClassification.from_pretrained(settings.lm_weights_path)
    tokenizer = AutoTokenizer.from_pretrained(settings.lm_weights_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    meta_df = pd.read_csv(settings.text_dataset_path)

    texts = meta_df["text"].tolist()
    logits = []  # 얘 사실은 prob임
    with torch.no_grad():
        for text in tqdm(texts):
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=settings.prev_seq_len,
                padding="max_length",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logit = outputs.logits
            prob = F.softmax(logit, dim=-1)[0, 1].item()
            logits.append(prob)

    probs = pd.Series(logits, index=meta_df.index)

    meta_df["logit"] = probs

    meta_df.to_csv(
        settings.text_dataset_path.with_name(settings.text_dataset_path.stem + "_logits.csv"), index=False
    )


if __name__ == "__main__":
    main()
