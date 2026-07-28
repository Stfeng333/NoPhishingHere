#this script implements a decision tree classifier that uses DistilBERT embeddings, TF-IDF features, and tabular features to classify emails as spam or not spam. It evaluates both a pure DistilBERT approach and a fused approach using 5-fold stratified cross-validation.

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBClassifier

# 1. loading Preprocessed Data
csv_path = Path.cwd() / "CEAS_08_cleaned.csv"
if not csv_path.exists():
    csv_path = Path.cwd().parent / "CEAS_08_cleaned.csv"

df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
y = df["label"].astype(int).values

# parsing Tabular Features
tabular_cols = [
    "subject_capitalized_percentage",
    "body_capitalized_percentage",
    "body_word_count",
    "subject_word_count",
    "is_weekend",
    "is_night",
    "is_known_domain",
    "is_known_fortune_domain",
    "total_char_count",
    "symbol_count",
    "symbol_ratio",
    "repeating_symbol_count",
]
X_tabular_raw = (
    df[tabular_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
)

# 2. Generate DistilBERT Embeddings
print("Extracting DistilBERT embeddings...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
model.eval()


def extract_bert_features(text_series, batch_size=32):
    text_list = text_series.tolist()
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_text = text_list[i : i + batch_size]
        inputs = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
    return np.vstack(embeddings)


X_bert = extract_bert_features(df["email_text_clean"])

# 3. function to evaluate either pure BERT or fused features using 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def evaluate_feature_set(feature_type="fused"):
    """Evaluates either 'pure_bert' or 'fused' using 5-Fold Stratified CV."""
    print(f"\n==========================================")
    print(f" Running Evaluation: {feature_type.upper()}")
    print(f"==========================================")

    oof_probs = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
        y_train, y_val = y[train_idx], y[val_idx]

        if feature_type == "pure_bert":
            X_train = X_bert[train_idx]
            X_val = X_bert[val_idx]
        else:  # Fused: Tabular + TF-IDF + BERT
            # Fit Scaler and TF-IDF ONLY on training fold to prevent leakage
            scaler = StandardScaler()
            tabular_train = scaler.fit_transform(X_tabular_raw[train_idx])
            tabular_val = scaler.transform(X_tabular_raw[val_idx])

            tfidf = TfidfVectorizer(
                ngram_range=(1, 3), max_features=1000, stop_words="english"
            )
            tfidf_train = tfidf.fit_transform(
                df["email_text_clean"].iloc[train_idx]
            )
            tfidf_val = tfidf.transform(df["email_text_clean"].iloc[val_idx])

            bert_train = csr_matrix(X_bert[train_idx])
            bert_val = csr_matrix(X_bert[val_idx])

            X_train = hstack(
                [csr_matrix(tabular_train), tfidf_train, bert_train]
            ).tocsr()
            X_val = hstack(
                [csr_matrix(tabular_val), tfidf_val, bert_val]
            ).tocsr()

        # the training classifier
        clf = XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]

    # let's use a threshold of 0.5 to convert probabilities to binary predictions
    preds_binary = (oof_probs >= 0.5).astype(int)

    # components of confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, preds_binary).ravel()
    fpr = fp / (fp + tn)

    # summary of the performance metrics
    print("\n--- Performance Metrics ---")
    print(f"Accuracy:  {accuracy_score(y, preds_binary):.4f}")
    print(f"Precision: {precision_score(y, preds_binary):.4f}")
    print(f"Recall:    {recall_score(y, preds_binary):.4f}")
    print(f"F1 Score:  {f1_score(y, preds_binary):.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y, oof_probs):.4f}")

    inbox = (oof_probs < 0.30).sum()
    quarantine = ((oof_probs >= 0.30) & (oof_probs <= 0.75)).sum()
    spam = (oof_probs > 0.75).sum()

    print("\n--- Decision Boundary Triage Distribution ---")
    print(f"Inbox      (P < 0.30):      {inbox:6d} ({inbox / len(df):.2%})")
    print(
        f"Quarantine (0.30 <= P <= 0.75): {quarantine:6d} ({quarantine / len(df):.2%})"
    )
    print(f"Spam       (P > 0.75):      {spam:6d} ({spam / len(df):.2%})")

    return oof_probs


#evaluations for both approaches
bert_probs = evaluate_feature_set(feature_type="pure_bert")
fused_probs = evaluate_feature_set(feature_type="fused")

