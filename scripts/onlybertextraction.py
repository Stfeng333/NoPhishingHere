from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# ------------------------------------------------------------------
# 1. Load Preprocessed Data
# ------------------------------------------------------------------
csv_path = Path.cwd() / "CEAS_08_cleaned.csv"
if not csv_path.exists():
    csv_path = Path.cwd().parent / "CEAS_08_cleaned.csv"

print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
labels = df["label"].astype(int).values

# FIX: Chain .fillna("") to ensure true NaNs are also emptied
text_data = df["email_text_clean"].fillna("").replace("NaN", "").tolist()

# ------------------------------------------------------------------
# 2. Extract DistilBERT Embeddings (CPU Only)
# ------------------------------------------------------------------
print("Configuring environment for CPU inference...")
device = torch.device("cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
model.eval()


def extract_bert_features(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_text = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Extract [CLS] token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)

        if (i // batch_size) % 10 == 0:
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} emails...")

    return np.vstack(embeddings)


print("Extracting DistilBERT embeddings...")
X_bert = extract_bert_features(text_data)

# ------------------------------------------------------------------
# 3. Save Embeddings and Labels to Disk
# ------------------------------------------------------------------
output_path = Path.cwd() / "bert_embeddings.npz"
np.savez_compressed(output_path, embeddings=X_bert, labels=labels)
print(f"Successfully saved embeddings to '{output_path}' with shape: {X_bert.shape}")
