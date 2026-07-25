from pathlib import Path
import html
import re
import unicodedata

import pandas as pd

# load the dataset
csv_path = Path.cwd() / 'CEAS_08.csv'
if not csv_path.exists():
    csv_path = Path.cwd().parent / 'CEAS_08.csv'

# let's read everything as text first so missing-value normalization is explicit and consistent
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

print('Columns:')
print(df.columns.tolist())

# we want to normalize blanks to a string sentinel so later stages can treat them consistently.
df = df.apply(lambda column: column.map(lambda value: value.strip() if isinstance(value, str) else value))
df = df.replace({'': pd.NA}).fillna('NaN')

# cleanup rules:
# - empty sender entries become 'NaN'
# - malformed entries like 'username <>' become 'NaN'
# - bare email addresses remain unchanged

def clean_sender(value: str) -> str:
    if value is None or value == 'NaN':
        return 'NaN'

    sender = str(value).strip()
    if not sender:
        return 'NaN'

    if re.search(r'<\s*>\s*$', sender):
        return 'NaN'

    return sender


def fix_encoding_artifacts(text: str) -> str:
    replacements = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€“': '-',
        'â€”': '-',
        'Ã©': 'é',
        'Ã¨': 'è',
        'Ãª': 'ê',
        'Ã«': 'ë',
        'Ã ': 'à',
        'Ã¢': 'â',
        'Ã§': 'ç',
        'Ã®': 'î',
        'Ã¯': 'ï',
        'Ã´': 'ô',
        'Ã»': 'û',
        'Ã¹': 'ù',
        'Ã¶': 'ö',
        'Ã¼': 'ü',
        'Â©': '©',
        'Â®': '®',
        'Â£': '£',
        'Â': '',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def replace_symbol_noise(text: str) -> str:
    return re.sub(r'([^\w\s]){4,}', ' [SYMBOL_NOISE] ', text)


def clean_text(value: str) -> str:
    if value is None or value == 'NaN':
        return 'NaN'

    text = str(value)
    text = html.unescape(text)
    text = text.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text.replace('\\"', '"').replace("\\'", "'")
    text = unicodedata.normalize('NFKC', text)
    text = fix_encoding_artifacts(text)
    text = replace_symbol_noise(text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text.replace('[symbol_noise]', '[SYMBOL_NOISE]')

# my way of cleaning sender and deriving email-level text features (you can change this the way you want)
df['sender'] = df['sender'].map(clean_sender)
df['subject_clean'] = df['subject'].map(clean_text)
df['body_clean'] = df['body'].map(clean_text)

df['raw_email_text'] = (
    df['subject'].astype(str).where(df['subject'] != 'NaN', '')
    + ' '
    + df['body'].astype(str).where(df['body'] != 'NaN', '')
).str.strip()

df['total_char_count'] = df['raw_email_text'].str.len().fillna(0).astype(int)
df['symbol_count'] = df['raw_email_text'].fillna('').str.count(r'[^\w\s]')
df['symbol_ratio'] = df['symbol_count'].div(df['total_char_count'].replace(0, pd.NA)).fillna(0)
df['repeating_symbol_count'] = df['raw_email_text'].fillna('').str.count(r'([^\w\s])\1{2,}')

# we will replace long runs of symbols in the cleaned text
for column in ['subject_clean', 'body_clean']:
    df[column] = (
        df[column]
        .replace('NaN', '')
        .str.replace(r'([^\w\s]){4,}', ' [SYMBOL_NOISE] ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.lower()
        .str.replace('[symbol_noise]', '[SYMBOL_NOISE]', regex=False)
        .replace('', 'NaN')
    )

# Let's have a feature of merged model-ready field for later vectorization.
df['email_text_clean'] = (
    df['subject_clean'].replace('NaN', '')
    + ' '
    + df['body_clean'].replace('NaN', '')
).str.strip()
df['email_text_clean'] = df['email_text_clean'].replace('', 'NaN')


print(df[['sender', 'subject_clean', 'body_clean', 'symbol_count', 'symbol_ratio', 'repeating_symbol_count']].head(20))
df.head(20)

