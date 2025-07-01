import pandas as pd
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

import torch
import random
import time
import gc

from sklearn.metrics.pairwise import cosine_similarity

# For text cleaning
import re

INPUT_FILE_PATH = "data/reddit_sentiment_scored.parquet"

OUTPUT_FILE_PATH_CSV = "data/reddit_topic_modeled.csv"
OUTPUT_FILE_PATH_PARQUET = "data/reddit_topic_modeled.parquet"
METRICS_OUTPUT_CSV = "data/reddit_metrics.csv"
METRICS_OUTPUT_PARQUET = "data/reddit_metrics.parquet"

RANDOM_STATE = 12345
BATCH_SIZE = 32

# Fix seeds for reproductivity
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)

try:
    df = pd.read_parquet(INPUT_FILE_PATH)
except FileNotFoundError:
    print("File not found.")
except Exception as e:
    print("Error:", e)

def calculate_metrics(topics, embeddings):
    try:
        topics = np.array(topics)
        embeddings = np.array(embeddings)

        n_total = len(topics)
        n_noise = np.sum(topics == -1)
        noise_ratio = n_noise / n_total if n_total > 0 else 0

        valid_topics = np.unique(topics[topics != -1])
        if len(valid_topics) == 0:
            return {
                "n_topics": 0,
                "noise_ratio": noise_ratio,
                "avg_topic_size": 0,
                "topic_size_std": 0,
                "largest_topic_size": 0,
                "smallest_topic_size": 0,
                "topic_coherence": 0,
                "mean_inter_topic_similarity": 0,
                "score": 0
            }

        # Compute sizes
        topic_sizes = [np.sum(topics == t) for t in valid_topics]
        avg_size = np.mean(topic_sizes)
        size_std = np.std(topic_sizes)
        max_size = np.max(topic_sizes)
        min_size = np.min(topic_sizes)

        # coherence inside topics (cosine sim between docs)
        coherences = []
        centroids = []

        for t in valid_topics:
            group = embeddings[topics == t]
            if len(group) > 1:
                sim = cosine_similarity(group)
                upper_vals = sim[np.triu_indices(len(group), k=1)]
                coherences.append(np.mean(upper_vals))
                centroids.append(np.mean(group, axis=0))
            elif len(group) == 1:
                centroids.append(group[0])

        avg_coh = np.mean(coherences) if coherences else 0

        # inter topic similarity (cosine between centroids)
        if len(centroids) > 1:
            centroids = np.array(centroids)
            sim = cosine_similarity(centroids)
            inter_sim = np.mean(sim[np.triu_indices(len(centroids), k=1)])
        else:
            inter_sim = 0

        final_score = avg_coh - inter_sim

        return {
            "n_topics": len(valid_topics),
            "noise_ratio": noise_ratio,
            "avg_topic_size": avg_size,
            "topic_size_std": size_std,
            "largest_topic_size": max_size,
            "smallest_topic_size": min_size,
            "topic_coherence": avg_coh,
            "mean_inter_topic_similarity": inter_sim,
            "score": final_score
        }

    except Exception as err:
        print("Metric computation failed:", err)
        return {
            "n_topics": -1,
            "noise_ratio": -1,
            "avg_topic_size": -1,
            "topic_size_std": -1,
            "largest_topic_size": -1,
            "smallest_topic_size": -1,
            "topic_coherence": -1,
            "mean_inter_topic_similarity": -1,
            "score": -1,
            "time_seconds": -1
        }


subreddits = df["subreddit"].unique()

reddit_customs_stop_words = [
    'like', 'reddit', 'thread', 'know', 'way', 'gt', 'relatively', 'org', 'make', 'removed', 'r', 'look', 'just', 'me',
    'automoderator', 'get', 'ive', 'wa', 'reading', 'doesn', 'deleted', 'thing', 'work', 'also', 'subreddit',
    'used', 'lot', 'message', 'im', 're', 'll', 'isn', 'using', 'see', 'safe', 'comment', 'good', 'better', 'u',
    'isnt', 'its', 'my', 'tell', 'looking', 'type', 'amp', 'post', 'ask', 'moderators', 'http', 'https', 'com',
    'dont', "doesn't", "it's", 'i', "don't", 've', 'trying', 'really', 'willing', 'far', 'seen', 'bot', 'www',
    'check', 'thanks', 'sure', 'maybe', 'want', 'need', 'help', 'best', 'use', 'important', 'ok', 'okay', 'thank',
    'luck', 'thx', 'joke', 'bro', 'haha', 'dm', 'sent', 'great', 'nice', 'yeah', 'lol', 'appreciate', 'cool', 'sorry',
    'feedback', 'oh', 'karma', 'link', 'website', 'out', 'question', 'hi', 'hello', 'hey', 'info', 'anyone', 'someone',
    'things', 'stuff', 'got', 'getting', 'still', 'even', 'something', 'nothing', 'issue', 'problem', 'solution',
    'any', 'didn', 'wasn', 'hasn', 'don', 'ca', 'let', 'new', 'old', 'already', 'tried', 'anyway', 'comments', 'think',
    'questions', 'discussion', 'discussions', 'lounge', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
    'saturday', 'sunday', 'mon', 'tue', 'tues', 'wed', 'thu', 'thurs', 'fri', 'sat', 'sun', 'january', 'february',
    'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december','jan', 'feb',
    'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec', 'query', '2020', '2021','2022', '2023',
    '2024', '2025', '2026', '2027','2028', '2029', '2030', 'doing', 'having', 'going', 'probably', 'likely', 'actually',
    'does', 'op', 'would', 'could', 'should', 'can', 'will', 'much', 'many', 'very', 'little', 'less', 'more',
    'most', 'least', 'ever', 'never', 'always', 'often', 'seldom', 'sometimes', 'rarely', 'usually', 'definitely',
    'absolutely', 'exactly', 'perhaps', 'possibly', 'certainly', 'quite', 'too', 'enough', 'indeed',
    'instead', 'however', 'therefore', 'consequently', 'meanwhile', 'elsewhere', 'otherwise', 'unless', 'whether',
    'whereas', 'although', 'though', 'because', 'since', 'while', 'when', 'where', 'why', 'how', 'what', 'who',
    'whom', 'whose', 'which', 'and', 'but', 'or', 'nor', 'so', 'yet', 'for', 'about', 'above', 'across', 'after',
    'against', 'along', 'among', 'around', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between',
    'beyond', 'by', 'down', 'during', 'except', 'inside', 'into', 'near', 'off', 'on', 'onto', 'opposite', 'outside',
    'over', 'past', 'round', 'through', 'to', 'toward', 'under', 'up', 'upon', 'with', 'within', 'without',
    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'here', 'there', 'we', 'you', 'he', 'she', 'it', 'they',
    'them', 'their', 'his', 'her', 'its', 'our', 'your', 'myself', 'yourself', 'himself', 'herself', 'itself',
    'ourselves', 'yourselves', 'themselves', 'every', 'each', 'all', 'some', 'no', 'none', 'both', 'either',
    'neither', 'several', 'few', 'such', 'whatsoever', 'whatever', 'whoever', 'whichever',
    'whenever', 'wherever', 'accordingly', 'furthermore', 'moreover', 'nonetheless',
    'notwithstanding', 'subsequently', 'thus', 'hence', 'thereby', 'hereby', 'therein', 'wherein', 'whereby',
    'whereupon', 'everywhere', 'nowhere', 'somehow', 'anyhow', 'anywhere', 'else', 'once', 'twice', 'thrice',
    'formerly', 'latterly', 'presently', 'shortly', 'briefly', 'soon', 'immediately', 'ago', 'henceforth',
    'thereafter', 'upto', 'via', 'amongst', 'amid', 'eg', 'ie', 'etc', 'viz', 'cf', 'per', 'vs',
    'aka', 'fwiw', 'imo', 'irl', 'nsfw', 'oc', 'tldr', 'yolo', 'afaik', 'eli5', 'ikr', 'smh', 'fml', 'ftfy',
    'gtfo', 'hmu', 'idk', 'iykyk', 'lmao', 'lmfao', 'rofl', 'til', 'wbu', 'wyd', 'wym', 'yada', 'yadda', 'yall',
    'yeet', 'bruh', 'cmon', 'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'bout', 'cuz', 'lemme', 'gimme','edit',
    'upvote', 'welcome', 'helpful', 'helped', 'ooo', 'guys', 'ill', 'hopefully', 'googled', 'looked', 'kind', 'thought',
    'answer', 'enjoy', 'vacation', 'stay', 'nope', 'did', 'days', 'later', 'anyrting', 'able', 'read', 'happy', 'shit',
    'sht', 'fuck', 'fk', 'fkin', 'fucking', 'fking', 'shitty', 'bullshit', 'crap', 'fuk', 'wtf', 'damn', 'heck',
    'pissed', 'motherfucker', 'dick', 'stupid', 'dumbass', 'posting', 'yes', 'commenting', 'remember', 'sehel', 'elik',
    'ah', 'ty', 'ahh', 'ahhh', 'haha', 'hahah', 'hahahahah'
]

stopwords = list(ENGLISH_STOP_WORDS) + reddit_customs_stop_words

def normalize_text(text, custom_stopwords):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # delete URLs
    text = re.sub(r'u\/\w+', '', text) # delete users
    text = re.sub(r'\b\d{1,2}[-/.]\d{1,2}[-/.](19|20)\d{2}\b', '', text) # delete date format
    text = re.sub(r'(?<=\d),(?=\d)', '.', text) # change the , into . between numbers
    text = re.sub(r'\s+', ' ', text)  # Normalise multiple space
    return text.strip()

df['topic'] = None
df['count'] = None
df['name'] = None
df['representation'] = None
df['topic_prob'] = None
df['topic_prob_dist'] = None

model = SentenceTransformer("all-mpnet-base-v2")
print(torch.cuda.get_device_name(0))
model.eval() # for reproductivity

start_time = time.time()

all_metrics = []

for subreddit in subreddits:
    print(f"Processing {subreddit}")
    start_subreddit_time = time.time()
    data = df[df['subreddit'] == subreddit]
    subreddit_count_docs = len(data)
    print(subreddit_count_docs)

    texts = []
    row_indices = []
    timestamps = []
    classes = []

    for idx, row in data.iterrows():
        text = ""
        if row['content_type'] == 'post':
            text = (str(row['post_title']) + " " + str(row['post_body'])).strip()
        elif row['content_type'] == 'comment':
            text = str(row['comment_body']).strip()

        # Important here. If subreddit is under 1500 documents, we need to clean the text to have better topic modeling
        if pd.notna(text) and len(text) > 20:
            if subreddit_count_docs < 1500:
                clean_text = normalize_text(text, stopwords)
                if len(clean_text.split()) > 3:
                    texts.append(clean_text)
                else:
                    continue
            else:
                texts.append(text)

            row_indices.append(idx)
            timestamps.append(row['datetime'].strftime('%Y-%m') if pd.notna(row['datetime']) else None)
            classes.append(row['content_type'])


    if len(texts) < 50:
        continue

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, device=device, show_progress_bar=True)

    if subreddit_count_docs < 1500:
        print("less than 1500 documents")
        umap_model = UMAP(
            n_neighbors=20,
            n_components=10,
            min_dist=0.0,
            metric='cosine',
            low_memory=False,
            random_state=RANDOM_STATE
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=5,
            metric='euclidean',
            cluster_selection_method='leaf',
            prediction_data=True
        )

        vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1, 2), max_df=0.95)
        topic_model = BERTopic(
            embedding_model=model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            representation_model=KeyBERTInspired(random_state=RANDOM_STATE),
            verbose=True,
            calculate_probabilities=True,
            nr_topics='default'
        )

    elif subreddit_count_docs > 8000 :
        print("more than 8000 documents")
        umap_model = UMAP(
            n_neighbors=15,
            n_components=6,
            min_dist=0.0,
            metric='cosine',
            low_memory=False,
            random_state=RANDOM_STATE
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=5,
            metric='euclidean',
            cluster_selection_method='leaf',
            prediction_data=True
        )

        vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1, 2), max_df=0.95)
        topic_model = BERTopic(
            embedding_model=model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            representation_model=KeyBERTInspired(random_state=RANDOM_STATE),
            verbose=True,
            calculate_probabilities=True,
            nr_topics='auto'
        )
    else :
        print("between 1500 and 8000 documents")
        umap_model = UMAP(
            n_neighbors=20,
            n_components=10,
            min_dist=0.0,
            metric='cosine',
            low_memory=False,
            random_state=RANDOM_STATE
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=5,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1, 2), max_df=0.95)
        topic_model = BERTopic(
            embedding_model=model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            representation_model=KeyBERTInspired(random_state=RANDOM_STATE),
            verbose=True,
            calculate_probabilities=True,
            nr_topics='auto'
        )


    topics, probs = topic_model.fit_transform(texts)

    topic_info = topic_model.get_topic_info()

    topic_dict = topic_info.set_index("Topic").to_dict(orient="index")

    # Safety check for index alignment
    assert len(topics) == len(row_indices) == len(texts), \
        f"Length mismatch for {subreddit}: topics={len(topics)}, indices={len(row_indices)}, texts={len(texts)}"


    df.loc[row_indices, 'topic'] = topics
    df.loc[row_indices, 'topic_prob'] = [probs[i][topic] if topic != -1 else np.nan for i, topic in enumerate(topics)]
    df.loc[row_indices, 'topic_prob_dist'] = pd.Series(
        [p.tolist() if topic != -1 else [] for p, topic in zip(probs, topics)],
        index=row_indices
    )
    df.loc[row_indices, 'count'] = [topic_dict.get(t, {}).get("Count") for t in topics]
    df.loc[row_indices, 'name'] = [topic_dict.get(t, {}).get("Name") for t in topics]
    df.loc[row_indices, 'representation'] = [str(topic_dict.get(t, {}).get("Representation", [])) for t in topics]

    # Calculate metrics
    metrics = calculate_metrics(topics, embeddings)
    metrics["subreddit"] = subreddit

    end_subreddit_time = time.time()
    elapsed_subreddit_time = end_subreddit_time - start_subreddit_time
    metrics["time_seconds"] = round(elapsed_subreddit_time, 2)

    print(f"Finished {subreddit} in {elapsed_subreddit_time:.2f} seconds.")

    all_metrics.append(metrics)

    del topic_model
    del vectorizer
    del embeddings
    gc.collect()

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

print("Topics added successfully for the two files.")

metrics_df = pd.DataFrame(all_metrics)

try:
    print(f"Metrics saved to CSV: {METRICS_OUTPUT_CSV}")
    metrics_df.to_csv(METRICS_OUTPUT_CSV, index=False)
except Exception as e:
    print(f"Failed to save metrics CSV: {e}")

try:
    print(f"Metrics saved to Parquet: {METRICS_OUTPUT_PARQUET}")
    metrics_df.to_parquet(METRICS_OUTPUT_PARQUET, index=False)
except Exception as e:
    print(f"Failed to save metrics Parquet: {e}")

print("Metrics successsfully saved.")