import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# 1. Load Data
csv_path = Path.cwd() / "CEAS_08_cleaned.csv"
if not csv_path.exists():
    csv_path = Path.cwd().parent / "CEAS_08_cleaned.csv"

df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
y = df['label'].astype(int).values

# 2. Setup Device & Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
model.eval()

# 3. Extract pure DistilBERT embeddings
def get_bert_embeddings(text_series, batch_size=32):
    text_list = text_series.tolist()
    embeddings = []
    
    for i in range(0, len(text_list), batch_size):
        batch_text = text_list[i:i + batch_size]
        inputs = tokenizer(
            batch_text, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token vector (768 dimensions)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
            
    return np.vstack(embeddings)

print("Extracting DistilBERT embeddings (No N-Grams)...")
X_bert = get_bert_embeddings(df['email_text_clean'])
print(f"Feature Matrix Shape (BERT only): {X_bert.shape}")

# 4. Evaluate using 5-Fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_bert, y)):
    X_train, y_train = X_bert[train_idx], y[train_idx]
    X_val, y_val = X_bert[val_idx], y[val_idx]
    
    clf = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        random_state=42,
        eval_metric='logloss'
    )
    clf.fit(X_train, y_train)
    
    oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
    print(f"Fold {fold + 1} complete.")

# 5. Output Metrics & Triage Statistics
binary_preds = (oof_preds >= 0.5).astype(int)
print("\n=== Pure DistilBERT Classification Report ===")
print(classification_report(y, binary_preds, digits=4))
print(f"ROC-AUC Score: {roc_auc_score(y, oof_preds):.4f}")

# Triage Distribution based on confidence scores
inbox_cnt = (oof_preds < 0.30).sum()
quarantine_cnt = ((oof_preds >= 0.30) & (oof_preds <= 0.75)).sum()
spam_cnt = (oof_preds > 0.75).sum()

print("\n=== Email Triage Distribution ===")
print(f"Inbox   (P < 0.30):      {inbox_cnt} ({inbox_cnt/len(df):.2%})")
print(f"Quarantine (0.30-0.75): {quarantine_cnt} ({quarantine_cnt/len(df):.2%})")
print(f"Spam    (P > 0.75):      {spam_cnt} ({spam_cnt/len(df):.2%})")

