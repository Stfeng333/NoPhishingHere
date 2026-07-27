#this is the script that will extract the features from the dataset and fuse them together into a single feature matrix (BERT + N-grams + Tabular features) for use in the decision tree classifier
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from scipy.sparse import save_npz

# 1. Load Data
csv_path = Path.cwd() / "CEAS_08_cleaned.csv"
if not csv_path.exists():
    csv_path = Path.cwd().parent / "CEAS_08_cleaned.csv"

df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

# Target vector
y = df['label'].astype(int).values

# 2. Extract DistilBERT Embeddings
print("Extracting DistilBERT embeddings...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
model.eval()

def get_bert_embeddings(text_list, batch_size=32):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_text = text_list[i:i + batch_size].tolist()
        inputs = tokenizer(batch_text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token representation (index 0)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
    return np.vstack(embeddings)

bert_features = get_bert_embeddings(df['email_text_clean'])

# 3. Tabular Features
tabular_cols = [
    'subject_capitalized_percentage', 'body_capitalized_percentage',
    'body_word_count', 'subject_word_count', 'is_weekend', 'is_night',
    'is_known_domain', 'is_known_fortune_domain', 'total_char_count',
    'symbol_count', 'symbol_ratio', 'repeating_symbol_count'
]
df_num = df[tabular_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
scaler = StandardScaler()
tabular_features = scaler.fit_transform(df_num)

# 4. N-Gram TF-IDF Features
print("Extracting TF-IDF N-grams...")
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=1000, stop_words='english')
tfidf_features = tfidf.fit_transform(df['email_text_clean'])

# 5. Combine All Features
print("Fusing feature representations...")
X_fused = hstack([csr_matrix(tabular_features), tfidf_features, csr_matrix(bert_features)]).tocsr()

print(f"Final Feature Matrix Shape: {X_fused.shape}")

#now we will save the sparse feature matrix
save_npz("x_fused_features.npz", X_fused)
np.save("y_labels.npy", y)
print ("Saved fused feature matrix and labels to disk.")

