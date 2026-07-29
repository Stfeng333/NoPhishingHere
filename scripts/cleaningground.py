from pathlib import Path
import html
import re
import unicodedata
import pandas as pd

# loading desired datasets
csv_path = Path.cwd() / "SpamAssasin.csv"
if not csv_path.exists():
    csv_path = Path.cwd().parent / "SpamAssasin.csv"

domain_csv_path = Path.cwd() / "top500Domains.csv"
if not domain_csv_path.exists():
    domain_csv_path = Path.cwd().parent / "top500Domains.csv"

domainfortune_csv_path = Path.cwd() / "Fortune_500_Email_Domains.csv"
if not domainfortune_csv_path.exists():
    domainfortune_csv_path = Path.cwd().parent / "Fortune_500_Email_Domains.csv"

domain_df = pd.read_csv(domain_csv_path)
fortune_df = pd.read_csv(domainfortune_csv_path)

df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

# drop receiver column if exists, this feature is not needed for the model
if "receiver" in df.columns:
    df = df.drop(columns=["receiver"])

original_count = len(df)

# drop duplicate email logs
df = df.drop_duplicates(subset=["subject", "body"], keep="first").reset_index(drop=True)
dropped_count = original_count - len(df)
print(f"Dropped {dropped_count} duplicate entries based on subject and body.")


# clean sender field to handle missing values and empty strings
def clean_sender(value: str) -> str:
    if not value or value == "NaN":
        return "NaN"
    sender = str(value).strip()
    if not sender or re.search(r"<\s*>\s*$", sender):
        return "NaN"
    return sender


df["sender"] = df["sender"].apply(clean_sender)


# we want to extract email, domain, and TLD from cleaned sender
def extract_email(sender):
    if sender == "NaN":
        return "NaN"
    match = re.search(r"<([^<>]+)>", sender)
    if match:
        return match.group(1).strip()
    return sender.strip()


df["sender_email"] = df["sender"].apply(extract_email)
df["sender_domain"] = df["sender_email"].str.split("@").str[1].fillna("NaN")
df["sender_tld"] = df["sender_domain"].str.split(".").str[-1].fillna("NaN")

# getting the capitalized percentage in subject and body
df["subject_capitalized_percentage"] = (
    df["subject"]
    .fillna("")
    .apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0.0)
)
df["body_capitalized_percentage"] = (
    df["body"]
    .fillna("")
    .apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0.0)
)

# counting body and subject word count
df["body_word_count"] = df["body"].str.split().str.len().fillna(0).astype(int)
df["subject_word_count"] = df["subject"].str.split().str.len().fillna(0).astype(int)

# preprocessing the dates
df["date_cleaned"] = (
    df["date"].astype(str).str.replace(r"^[A-Za-z]{3},\s+", "", regex=True)
)
df["date_cleaned"] = df["date_cleaned"].str.replace(r"\s+\([^)]+\)$", "", regex=True)

df["datetime"] = pd.to_datetime(
    df["date_cleaned"], format="%d %b %Y %H:%M:%S %z", errors="coerce", utc=True
)
df["day_of_week"] = df["datetime"].dt.day_name()
df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
df["is_night"] = (df["datetime"].dt.hour < 5).fillna(0).astype(int)

df = df.drop(columns=["date_cleaned", "date"], errors="ignore")


# Text cleaning helpers
def fix_encoding_artifacts(text: str) -> str:
    replacements = {
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "Ã©": "é",
        "Ã¨": "è",
        "Ãª": "ê",
        "Ã«": "ë",
        "Ã": "à",
        "Ã¢": "â",
        "Ã§": "ç",
        "Ã®": "î",
        "Ã¯": "ï",
        "Ã´": "ô",
        "Ã»": "û",
        "Ã¹": "ù",
        "Ã¶": "ö",
        "Ã¼": "ü",
        "Â©": "©",
        "Â®": "®",
        "Â£": "£",
        "Â": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def replace_symbol_noise(text: str) -> str:
    return re.sub(r"([^\w\s]){4,}", " [SYMBOL_NOISE] ", text)


def clean_text(value: str) -> str:
    if not value or value == "NaN":
        return "NaN"
    text = str(value)
    text = html.unescape(text)
    text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = text.replace('\\"', '"').replace("\\'", "'")
    text = unicodedata.normalize("NFKC", text)
    text = fix_encoding_artifacts(text)
    text = replace_symbol_noise(text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = text.replace("[symbol_noise]", "[SYMBOL_NOISE]")
    return text if text else "NaN"


df["subject_clean"] = df["subject"].apply(clean_text)
df["body_clean"] = df["body"].apply(clean_text)

# known domain flags
df["is_known_domain"] = df["sender_domain"].isin(domain_df["Root Domain"]).astype(int)
df["is_known_fortune_domain"] = (
    df["sender_domain"].isin(fortune_df["Domain"]).astype(int)
)

# joining the subject and body to create a single email text field for comparison
df["raw_email_text"] = (
    df["subject"].astype(str).replace("NaN", "")
    + " "
    + df["body"].astype(str).replace("NaN", "")
).str.strip()

# counting total characters, symbols, symbol ratio, and repeating symbols
df["total_char_count"] = df["raw_email_text"].str.len().fillna(0).astype(int)
df["symbol_count"] = df["raw_email_text"].fillna("").str.count(r"[^\w\s]")
df["symbol_ratio"] = (
    df["symbol_count"].div(df["total_char_count"].replace(0, pd.NA)).fillna(0.0)
)
df["repeating_symbol_count"] = (
    df["raw_email_text"].fillna("").str.count(r"([^\w\s])\1{2,}")
)

# joining cleaned subject and body to create a single email text field
df["email_text_clean"] = (
    df["subject_clean"].replace("NaN", "") + " " + df["body_clean"].replace("NaN", "")
).str.strip()
df["email_text_clean"] = df["email_text_clean"].replace("", "NaN")

if "urls" not in df.columns:
    df["urls"] = "NaN"

# chosen features for export (can change depending on your choice of features)
export_cols = [
    "subject_clean",
    "body_clean",
    "subject_capitalized_percentage",
    "body_capitalized_percentage",
    "body_word_count",
    "subject_word_count",
    "sender_tld",
    "is_weekend",
    "is_night",
    "is_known_domain",
    "is_known_fortune_domain",
    "total_char_count",
    "symbol_count",
    "symbol_ratio",
    "repeating_symbol_count",
    "email_text_clean",
    "urls",
    "label",
]

#name the new dataset file as you like
df = df[export_cols]
df.to_csv("cleanassassin.csv", index=False)

