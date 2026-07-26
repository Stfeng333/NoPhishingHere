from pathlib import Path
import html
import re
import unicodedata
import nltk
from nltk.util import ngrams
from nltk.tokenize import wordpunct_tokenize

import pandas as pd

# load the dataset
csv_path = Path.cwd() / 'CEAS_08_cleaned.csv'
if not csv_path.exists():
    csv_path = Path.cwd().parent / 'CEAS_08_cleaned.csv'
    
#extract the ngrams from the subject and body columns
def extract_ngrams(text, n):
    if pd.isna(text):
        return []
    # Tokenize without requiring external NLTK corpora.
    tokens = wordpunct_tokenize(text)
    # Generate n-grams
    n_grams = list(ngrams(tokens, n))
    # Join the n-grams into strings
    return [' '.join(gram) for gram in n_grams]


