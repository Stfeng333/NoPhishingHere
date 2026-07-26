from pathlib import Path
import html
import re
import unicodedata

import pandas as pd

# load the dataset
csv_path = Path.cwd() / 'CEAS_08.csv'
if not csv_path.exists():
    csv_path = Path.cwd().parent / 'CEAS_08.csv'

domain_csv_path = Path.cwd() / 'top500Domains.csv'
if not domain_csv_path.exists():
    domain_csv_path = Path.cwd().parent / 'top500Domains.csv'
    
domainfortune_csv_path = Path.cwd() / 'Fortune_500_Email_Domains.csv'
if not domainfortune_csv_path.exists():
    domainfortune_csv_path = Path.cwd().parent / 'Fortune_500_Email_Domains.csv'

domain_df = pd.read_csv(domain_csv_path)
fortune_df = pd.read_csv(domainfortune_csv_path)

# let's read everything as text first so missing-value normalization is explicit and consistent
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)


#drop receiver column
df = df.drop(columns=['receiver'])

original_count = len(df)

# Count the number of entries dropped
df = df.drop_duplicates(subset=['subject', 'body'], keep='first')
dropped_count = original_count - len(df)
print(f"Dropped {dropped_count} duplicate entries based on subject and body.")

#counting capiltal letters in the subject and body
df['subject_capitalized_percentage'] = df['subject'].fillna('').apply(lambda x: sum(1 for c in x if c.isupper())/len(x) if len(x) > 0 else 0)
df['body_capitalized_percentage'] = df['body'].fillna('').apply(lambda x: sum(1 for c in x if c.isupper())/len(x) if len(x) > 0 else 0)

#body and subject word count
df['body_word_count'] = df['body'].str.split().str.len()
df['subject_word_count'] = df['subject'].str.split().str.len()

#extraction of sender domain, and sender tld
#sender email is set to the email if there is no name and the email is without the "<>" brackets, otherwise it is set to the email inside the "<>" brackets
def extract_email(sender):
    if pd.isna(sender):
        return None
    match = re.search(r'<([^<>]+)>', sender)
    if match:
        return match.group(1).strip()
    else:
        return sender.strip()
df['sender_email'] = df['sender'].apply(extract_email)
df['sender_domain'] = df['sender_email'].str.split('@').str[1]
df['sender_tld'] = df['sender_domain'].str.split('.').str[-1]

df['date_cleaned'] = df['date'].astype(str).str.replace(r'^[A-Za-z]{3},\s+', '', regex=True) # Remove day of the week (e.g., "Wed, ")
df['date_cleaned'] = df['date_cleaned'].str.replace(r'\s+\([^)]+\)$', '', regex=True) # Remove text in parentheses (e.g., "(UTC)")

df['datetime'] = pd.to_datetime(df['date_cleaned'], format='%d %b %Y %H:%M:%S %z', errors='coerce', utc=True)
df['day_of_week'] = df['datetime'].dt.day_name()

# Drop the temporary 'date_cleaned' column
df = df.drop(columns=['date_cleaned'])

df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
df['is_night'] = (df['datetime'].dt.hour < 5).astype(int)

#drop original date column
df.drop(columns=['date'], inplace=True)

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

#feature engineering to check if the sender domain belongs to the top 500 domains or fortune 500 domains
df['is_known_domain'] = df['sender_domain'].isin(domain_df['Root Domain']).astype(int)
df['is_known_fortune_domain'] = df['sender_domain'].isin(fortune_df['Domain']).astype(int)



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

# create a new cleaned CSV file with the processed data with the following columns: , subject_clean, body_clean, subject_capitalized_percentage, body_capitalized_percentage, body_word_count, subject_word_count, sender_tld, is_weekend, is_night, is_known_domain, is_known_fortune_domain, total_char_count, symbol_count, symbol_ratio, repeating_symbol_count, email
df = df[['subject_clean', 'body_clean', 'subject_capitalized_percentage', 'body_capitalized_percentage', 'body_word_count', 'subject_word_count', 'sender_tld', 'is_weekend', 'is_night', 'is_known_domain', 'is_known_fortune_domain', 'total_char_count', 'symbol_count', 'symbol_ratio', 'repeating_symbol_count', 'email_text_clean']]

df.to_csv('CEAS_08_cleaned.csv', index=False)
