import pandas as pd

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import time

INPUT_FILE_PATH = "data/reddit_data.parquet"

OUTPUT_FILE_PATH_CSV = "data/reddit_sentiment_scored.csv"
OUTPUT_FILE_PATH_PARQUET = "data/reddit_sentiment_scored.parquet"



try:
    df = pd.read_parquet(INPUT_FILE_PATH)
except FileNotFoundError:
    print("File not found.")
except Exception as e:
    print("Error:", e)

# Add technology and reddit specific words
tech_sentiment_words = {
    # Negative words
    'bug': -2.0,
    'deprecated': -1.5,
    'crash': -3.0,
    'laggy': -2.0,
    'lag': -2.0,
    'unstable': -2.5,
    'vulnerability': -3.0,
    'backdoor': -3.0,
    'broken': -2.0,
    'unusable': -2.5,
    'hacky': -1.5,
    'hack': -1.5,
    'workaround': -1.0,
    'downtime': -2.0,
    'exploit': -2.5,
    'scam': -4.0,
    'bottleneck': -1.5,

    # Positive words
    'innovative': 2.5,
    'efficient': 2.0,
    'optimized': 2.0,
    'trendy': 2.0,
    'intuitive': 2.0,
    'elegant': 2.5,
    'seamless': 2.5,
    'breakthrough': 3.0,
    'robust': 2.0,
    'scalable': 1.5,
    'powerful': 2.0,
    'user-friendly': 2.5,
    'cutting-edge': 2.0,
    'state-of-the-art': 2.5,
    'performant': 2.0,
    'reliable': 2.0,
    'responsive': 1.5,
    'revolutionary': 3.0,
}

nltk.download("vader_lexicon", quiet=True)

sentiment_analyzer = SentimentIntensityAnalyzer()
sentiment_analyzer.lexicon.update(tech_sentiment_words)

# Sentiment function: if no text = neutral, if text too short like thank you, nice = neutral, else apply Vader
def get_sentiment(text):
    if not isinstance(text, str):
        return 0.0
    elif len(text.strip()) < 4:
        return 0.0
    else:
        sentiment = sentiment_analyzer.polarity_scores(text)
        return sentiment["compound"]

posts_df = df[df['content_type'] == 'post'].copy()

# Apply sentiment on post (title + body)
def get_post_text_sentiment(row):
    if pd.notna(row["post_title"]):
        title = str(row["post_title"])
    else:
        title = ""

    if pd.notna(row["post_body"]):
        body = str(row["post_body"])
    else:
        body = ""

    full_text = title + " " + body
    full_text = full_text.strip()  # No extra spaces

    return get_sentiment(full_text)

start_time = time.time()

# Sentiment for posts only
posts_df["post_sentiment"] = posts_df.apply(get_post_text_sentiment, axis=1)

# Mapping of post_id to post_sentiment
post_sentiment_map = posts_df.set_index('post_id')['post_sentiment'].to_dict()

# Apply post sentiment to all rows based on post_id
df["post_sentiment"] = df["post_id"].map(post_sentiment_map)

# Calculate comment sentiment for all rows
df["comment_sentiment"] = df["comment_body"].apply(get_sentiment)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Topic modeling completed in {elapsed_time:.2f} seconds.")


try:
    print(f"Enriched dataset saved to CSV: {OUTPUT_FILE_PATH_CSV}")
    df.to_csv(OUTPUT_FILE_PATH_CSV, index=False)
except Exception as e:
    print(f"Failed to save CSV: {e}")

try:
    print(f"Enriched dataset saved to Parquet: {OUTPUT_FILE_PATH_PARQUET}")
    df.to_parquet(OUTPUT_FILE_PATH_PARQUET, index=False)
except Exception as e:
    print(f"Failed to save Parquet: {e}")

print("Vader added successfully for the two files.")