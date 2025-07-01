# Reddit technology monitoring – master's thesis

This repository contains the code and data pipeline for my master's thesis project, which investigates how emerging technologies are discussed on Reddit. The goal is to extract public sentiment, semantic structures, and trends from technology-focused subreddits.

## Project overview

The analysis involves three main components:

- **Sentiment analysis** using VADER (Valence Aware Dictionary for sEntiment Reasoning)
- **Topic modeling** using BERTopic (with MPNet embeddings, UMAP, HDBSCAN, and KeyBERTInspired representation)
- **Trend analysis**

## Tools and technologies

- Python 3.11
- `asyncpraw` for asynchronous Reddit data collection
- `vaderSentiment` for sentiment scoring
- `BERTopic` for topic modeling
- `sentence-transformers`, `umap-learn`, `hdbscan`, `scikit-learn` for embeddings and clustering
- `pandas`, `numpy`, `matplotlib` for data handling and visualization

## Installation and usage

### 1. Install required libraries

Run the following command to install all necessary dependencies:

```bash
pip install pandas asyncpraw asyncprawcore nltk sentence-transformers scikit-learn umap-learn hdbscan bertopic torch
```

---

### 2. Configure Reddit API credentials

Create a file named `config.py` in the root directory with your Reddit API credentials:

```python
# config.py
CLIENT_ID = "your_client_id_here"
CLIENT_SECRET = "your_client_secret_here"
USER_AGENT = "your_user_agent_here"
```

---

### 3. Run the pipeline

Execute the following scripts in order:

#### Step 1: `fetch_reddit.py`  
Collects Reddit posts and comments asynchronously.  
*This step will take several hours.*

#### Step 2: `sentiment_analysis.py`  
Performs sentiment scoring on all documents using VADER.

#### Step 3: `topic_modeling.py`  
Applies BERTopic to generate and analyze discussion topics.

---

### 4. Data

All input and output datasets are saved in the `data/` directory.  
Final datasets include enriched information such as sentiment scores and topic assignments, available in both `.parquet` and `.csv` formats.

---

### 6. Testing

A set of experimental scripts is available in the `/test/` directory. These were used to evaluate and compare multiple BERTopic configurations across different subreddits.  
Evaluation metrics include intra-topic coherence, inter-topic similarity, and the proportion of noise.

---

### 5. License

This project was developed as part of an academic research project.  
Reuse for academic purposes is permitted with proper citation.
