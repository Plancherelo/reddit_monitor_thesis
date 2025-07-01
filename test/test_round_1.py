import os
import pandas as pd
import gc
import time
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import random
import torch
import re


# Configuration
INPUT_FILE = "../data/reddit_sentiment_scored.parquet"
SAVE_DIR = "tests_results_round_1/"
os.makedirs(SAVE_DIR, exist_ok=True)
RANDOM_STATE = 12345
BATCH_SIZE = 32

# Fix seeds for reproductivity
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)

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

# Model parameters
embedders = {
    "BAAI": 'BAAI/bge-base-en-v1.5',
    "e5-large-v2": "intfloat/e5-large-v2",
    "MiniLM": "all-MiniLM-L6-v2",
    "MPNet": "all-mpnet-base-v2"
}

reductions = {
    "UMAP_default": {"model": "umap", "n_neighbors": 15, "n_components": 2},
    "UMAP_middle_dim": {"model": "umap", "n_neighbors": 15, "n_components": 6},
    "UMAP_high_dim": {"model": "umap", "n_neighbors": 20, "n_components": 10},
}

cluster_options = {
    "clu_default": {"min_cluster_size": 5, "min_samples": None, "cluster_selection_method": "eom"},
    "clu_default_leaf": {"min_cluster_size": 5, "min_samples": None, "cluster_selection_method": "leaf"},
    "clu_medium": {"min_cluster_size": 15, "min_samples": 10, "cluster_selection_method": "eom"},
    "clu_heuristic": lambda n: {"min_cluster_size": max(10, min(50, int(0.015 * n))),"min_samples": max(3, min(20, int(0.008 * n))), "cluster_selection_method": "eom"},
}

topic_size_options = {"default": 10}

vectorizer_options = {
    "vec_bigram": {"ngram_range": (1, 2), "max_features": 5000}
}

nr_topics_options = ["default", "auto"]

def create_reduction_model(params):
    return UMAP(
        n_components=params["n_components"],
        n_neighbors=params["n_neighbors"],
        random_state=RANDOM_STATE,
        # low_memory=True,
        n_jobs=1,
        verbose=False
    )

def create_clustering_model(cluster_params):
    return HDBSCAN(
        min_cluster_size=cluster_params["min_cluster_size"],
        min_samples=cluster_params.get("min_samples"),
        cluster_selection_method=cluster_params.get("cluster_selection_method", "eom"),
        prediction_data=True
    )

def create_vectorizer_model(vec_params):
    return CountVectorizer(stop_words=stopwords, **vec_params)

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

def process_subreddit(subreddit, df):
    data = df[df['subreddit'] == subreddit]

    data.loc[:, "doc"] = np.where(
        data["content_type"] == "post",
        data["post_title"].fillna("") + " " + data["post_body"].fillna(""),
        data["comment_body"].fillna("")
    )
    docs = data["doc"].str.strip()
    docs = docs[docs.str.len() > 20].tolist()

    if len(docs) < 50:
        print(f"Subreddit {subreddit} ignored (not enough documents).")
        return

    print(f"Number of documents for {subreddit}: {len(docs)}")

    for embed_name, embed_model_name in embedders.items():
        print(f"Embedding model: {embed_name}")
        model = SentenceTransformer(embed_model_name)
        print(torch.cuda.get_device_name(0))
        model.eval()  # for reproductivity


        docs_input = [f"query: {doc}" for doc in docs] if embed_name == "e5-large-v2" else docs
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = model.encode(docs_input, batch_size=BATCH_SIZE, device=device, show_progress_bar=True)

        for reducer_name, reducer_params in reductions.items():
            print(f"Dimensionality reduction: {reducer_name}")

            for cluster_name, cluster_config in cluster_options.items():
                cluster_params = cluster_config(len(docs)) if callable(cluster_config) else cluster_config
                print(f"Clustering method: {cluster_name} (min_cluster_size={cluster_params['min_cluster_size']})")

                for vectorizer_name, vec_params in vectorizer_options.items():
                    for size_name, min_topic_size in topic_size_options.items():
                        for nr_topic in nr_topics_options:
                            config_str = f"{embed_name}-{reducer_name}-{cluster_name}-{size_name}-{vectorizer_name}-{nr_topic}"
                            print(f"Configuration: {config_str}")

                            try:
                                umap_model = create_reduction_model(reducer_params)
                                hdbscan_model = create_clustering_model(cluster_params)
                                vectorizer_model = create_vectorizer_model(vec_params)

                                nr = None if nr_topic == "default" else "auto"

                                representation_model = KeyBERTInspired(random_state=RANDOM_STATE)

                                topic_model = BERTopic(
                                    embedding_model=model,
                                    umap_model=umap_model,
                                    hdbscan_model=hdbscan_model,
                                    vectorizer_model=vectorizer_model,
                                    min_topic_size=min_topic_size,
                                    representation_model=representation_model,
                                    calculate_probabilities=True,
                                    verbose=False,
                                    nr_topics=nr,
                                )

                                start_time = time.time()
                                topics, probs = topic_model.fit_transform(docs_input, embeddings)
                                training_time = time.time() - start_time

                                topic_info = topic_model.get_topic_info()

                                topic_mapping = topic_info.set_index("Topic")[
                                    ["Name", "Count", "Representation"]].to_dict('index')

                                result_df = pd.DataFrame({
                                    "subreddit": subreddit,
                                    "text": docs,
                                    "topic": topics,
                                    "topic_prob": [probs[i][topic] if topic != -1 else np.nan for i, topic in enumerate(topics)],
                                    "topic_prob_dist": [list(p) for p in probs],
                                    "config_id": config_str,
                                    "topic_label": [topic_mapping.get(topic, {}).get("Name", "Outlier") for topic in topics],
                                    "topic_count": [topic_mapping.get(topic, {}).get("Count", 0) for topic in topics],
                                    "top_words": [topic_mapping.get(topic, {}).get("Representation", []) for topic in topics]
                                })
                                result_df.to_parquet(os.path.join(SAVE_DIR, f"{subreddit}__{config_str}_v6.parquet"), index=False)

                                metrics_start = time.time()

                                metrics = calculate_metrics(topics, embeddings)

                                metrics_time = time.time() - metrics_start

                                metrics["subreddit"] = subreddit
                                metrics["config_id"] = config_str
                                metrics["training_time"] = training_time
                                metrics["metrics_calculation_time"] = metrics_time

                                metrics_df = pd.DataFrame([metrics])
                                metrics_df.to_parquet(os.path.join(SAVE_DIR, f"{subreddit}__{config_str}_metrics_v6.parquet"), index=False)

                                print(f"Finished configuration {config_str} with {metrics['n_topics']} topics in {training_time:.1f} seconds")

                            except Exception as e:
                                print(f"Error during configuration {config_str}: {e}")
                            finally:
                                try:
                                    del umap_model, hdbscan_model, vectorizer_model, topic_model
                                except:
                                    pass
                                gc.collect()

        del embeddings
        del model
        gc.collect()


df = pd.read_parquet(INPUT_FILE)
subreddit_counts = df["subreddit"].value_counts()
selected_subreddits = [
    subreddit_counts.idxmax(),
    subreddit_counts.index[len(subreddit_counts) // 2],
    subreddit_counts.idxmin()
]

for subreddit in selected_subreddits:
    process_subreddit(subreddit, df)