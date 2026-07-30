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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBClassifier

# ==========================================
# 1. Load Preprocessed Dataset & Split 80/20
# ==========================================
csv_path = Path.cwd() / "CEAS_08_cleaned.csv"
if not csv_path.exists():
    csv_path = Path.cwd().parent / "CEAS_08_cleaned.csv"

df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

# Perform 80/20 Train/Test split using stratified sampling
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["label"]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

y_train = train_df["label"].astype(int).values
y_test = test_df["label"].astype(int).values

print(f"Total dataset size: {len(df)}")
print(f"Training set size (80%): {len(train_df)}")
print(f"Test set size (20%):     {len(test_df)}")

# Tabular features configuration
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

X_tab_train_raw = train_df[tabular_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
X_tab_test_raw = test_df[tabular_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

# Clean text data arrays
text_train = train_df["email_text_clean"].replace("NaN", "").values
text_test = test_df["email_text_clean"].replace("NaN", "").values

# ==========================================
# 2. Extract DistilBERT Embeddings
# ==========================================
print("\nExtracting DistilBERT embeddings...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
model.eval()

def extract_bert_features(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_text = texts[i : i + batch_size].tolist()
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

X_bert_train = extract_bert_features(text_train)
X_bert_test = extract_bert_features(text_test)

# ==========================================
# 3. Evaluation & Fusion Function
# ==========================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def print_metrics(y_true, y_probs, set_name="Test Set"):
    preds_binary = (y_probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds_binary).ravel()
    fpr = fp / (fp + tn)

    print(f"\n--- {set_name} Performance Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_true, preds_binary):.4f}")
    print(f"Precision: {precision_score(y_true, preds_binary):.4f}")
    print(f"Recall:    {recall_score(y_true, preds_binary):.4f}")
    print(f"F1 Score:  {f1_score(y_true, preds_binary):.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_probs):.4f}")

    inbox = (y_probs < 0.30).sum()
    quarantine = ((y_probs >= 0.30) & (y_probs <= 0.75)).sum()
    spam = (y_probs > 0.75).sum()
    total = len(y_true)

    print(f"\n--- {set_name} Decision Boundary Triage Distribution ---")
    print(f"Inbox------(P < 0.30):-------{inbox:6d} ({inbox / total:.2%})")
    print(f"Quarantine-(0.30 <= P <= 0.75): {quarantine:6d} ({quarantine / total:.2%})")
    print(f"Spam-------(P > 0.75):-------{spam:6d} ({spam / total:.2%})")

def evaluate_feature_set(feature_type="fused"):
    print(f"\n==========================================")
    print(f" Running Evaluation: {feature_type.upper()}")
    print(f"==========================================")

    # --- Step 3A: 5-Fold Cross Validation on 80% Train Set ---
    oof_probs = np.zeros(len(train_df))

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
        y_tr, y_va = y_train[trn_idx], y_train[val_idx]

        if feature_type == "pure_bert":
            X_tr = X_bert_train[trn_idx]
            X_va = X_bert_train[val_idx]

        elif feature_type == "pure_tfidf":
            tfidf = TfidfVectorizer(
                ngram_range=(1, 3), max_features=1000, stop_words="english"
            )
            X_tr = tfidf.fit_transform(text_train[trn_idx])
            X_va = tfidf.transform(text_train[val_idx])

        else:  # Fused
            scaler = StandardScaler()
            tab_tr = scaler.fit_transform(X_tab_train_raw[trn_idx])
            tab_va = scaler.transform(X_tab_train_raw[val_idx])

            tfidf = TfidfVectorizer(
                ngram_range=(1, 3), max_features=1000, stop_words="english"
            )
            tfidf_tr = tfidf.fit_transform(text_train[trn_idx])
            tfidf_va = tfidf.transform(text_train[val_idx])

            bert_tr = csr_matrix(X_bert_train[trn_idx])
            bert_va = csr_matrix(X_bert_train[val_idx])

            X_tr = hstack([csr_matrix(tab_tr), tfidf_tr, bert_tr]).tocsr()
            X_va = hstack([csr_matrix(tab_va), tfidf_va, bert_va]).tocsr()

        clf = XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)
        oof_probs[val_idx] = clf.predict_proba(X_va)[:, 1]

    print_metrics(y_train, oof_probs, set_name="5-Fold Cross-Validation (80% Train Set)")

    # --- Step 3B: Final Training on Full 80% Train Set & Testing on Held-out 20% Test Set ---
    if feature_type == "pure_bert":
        X_full_train = X_bert_train
        X_full_test = X_bert_test

    elif feature_type == "pure_tfidf":
        tfidf = TfidfVectorizer(
            ngram_range=(1, 3), max_features=1000, stop_words="english"
        )
        X_full_train = tfidf.fit_transform(text_train)
        X_full_test = tfidf.transform(text_test)

    else:  # Fused
        scaler = StandardScaler()
        tab_full_train = scaler.fit_transform(X_tab_train_raw)
        tab_full_test = scaler.transform(X_tab_test_raw)

        tfidf = TfidfVectorizer(
            ngram_range=(1, 3), max_features=1000, stop_words="english"
        )
        tfidf_full_train = tfidf.fit_transform(text_train)
        tfidf_full_test = tfidf.transform(text_test)

        bert_full_train = csr_matrix(X_bert_train)
        bert_full_test = csr_matrix(X_bert_test)

        X_full_train = hstack([csr_matrix(tab_full_train), tfidf_full_train, bert_full_train]).tocsr()
        X_full_test = hstack([csr_matrix(tab_full_test), tfidf_full_test, bert_full_test]).tocsr()

    clf = XGBClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=6,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    clf.fit(X_full_train, y_train)
    test_probs = clf.predict_proba(X_full_test)[:, 1]

    print_metrics(y_test, test_probs, set_name="Held-Out (20% Test Set)")

    return oof_probs, test_probs


# Main execution evaluating pure BERT, pure TF-IDF, and Fused features
bert_oof, bert_test = evaluate_feature_set(feature_type="pure_bert")
tfidf_oof, tfidf_test = evaluate_feature_set(feature_type="pure_tfidf")
fused_oof, fused_test = evaluate_feature_set(feature_type="fused")
