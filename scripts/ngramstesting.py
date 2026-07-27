from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import nltk
from nltk.util import ngrams
from pathlib import Path
import html
import re
import unicodedata


csv_path = Path.cwd() / "CEAS_08_cleaned.csv"
if not csv_path.exists():
    csv_path = Path.cwd().parent / "CEAS_08_cleaned.csv"


df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

# split into validation/test/train
X, X_val, y, y_val = train_test_split(df['body_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words='english',
    lowercase=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 6. Predict and Evaluate
predictions = model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}\n")
print("Classification Report:")
print(classification_report(y_test, predictions))

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('clf', LogisticRegression())
])

param_grid = {
    'tfidf__ngram_range': [(1, 2), (2, 3), (3, 4), (4, 5)]
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=kfold,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X, y)

print(f"Best N-gram Range: {grid_search.best_params_['tfidf__ngram_range']}")
print(f"Best CV Accuracy Score: {grid_search.best_score_:.3f}\n")

# View details for all tested combinations
print("All Grid Search Results:")
for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
    print(f"Mean Accuracy: {mean_score:.3f} using {params}")