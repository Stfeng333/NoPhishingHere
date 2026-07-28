# this script is only testing the n-gram features and tabular features without using BERT embeddings. It will evaluate the performance of a decision tree classifier using only these features.
# Henry's code


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

file = "CEAS_08_cleaned.csv"

df = pd.read_csv(file)

X, X_val, y, y_val = train_test_split(
    df["body_clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression()),
    ]
)

param_grid = {"tfidf__ngram_range": [(1, 2), (2, 3), (3, 4), (4, 5)]}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline, param_grid=param_grid, cv=kfold, scoring="accuracy", n_jobs=-1
)

grid_search.fit(X, y)

print(f"Best N-gram Range: {grid_search.best_params_['tfidf__ngram_range']}")
print(f"Best CV Accuracy Score: {grid_search.best_score_:.3f}\n")

# View details for all tested combinations
print("All Grid Search Results:")
for mean_score, params in zip(
    grid_search.cv_results_["mean_test_score"], grid_search.cv_results_["params"]
):
    print(f"Mean Accuracy: {mean_score:.3f} using {params}")
