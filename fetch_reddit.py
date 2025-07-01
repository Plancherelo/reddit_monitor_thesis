import pandas as pd

import asyncio
import asyncprawcore
import asyncpraw

import os
import time

from config import CLIENT_ID, CLIENT_SECRET, USER_AGENT

POST_LIMIT = 1000
COMMENTS_LIMIT = 20
NESTED_COMMENT_LIMIT = 15
DATA_DIR = "data"

OUTPUT_FILE_PATH_CSV = os.path.join(DATA_DIR, "reddit_data.csv")
OUTPUT_FILE_PATH_PARQUET = os.path.join(DATA_DIR, "reddit_data.parquet")

semaphore = asyncio.Semaphore(5) #limit concurrent tasks

SUBREDDITS = [
    "5Ginfo",
    "amateurradio",
    "analog",
    "antivirus",
    "artificial",
    "aerospaceengineering",
    "aviation",
    "batteries",
    "bigdata",
    "Cameras",
    "CloudSecurityPros",
    "compression",
    "cryptography",
    "cybersecurity",
    "Database",
    "datascience",
    "DDoSNetworking",
    "deeplearning",
    "drones",
    "electricalengineering",
    "engineering",
    "flightsim",
    "forensics",
    "Futurology",
    "gis",
    "hci",
    "Helicopters",
    "IOT",
    "LanguageTechnology",
    "labrats",
    "lasers",
    "lidar",
    "MachineLearning",
    "massspectrometry",
    "materials",
    "metallurgy",
    "netsec",
    "networking",
    "operatingsystems",
    "privacy",
    "QuantumComputing",
    "radar",
    "remotesensing",
    "rocketry",
    "robotics",
    "SDR",
    "SelfDrivingCars",
    "SelfSufficiency",
    "simracing",
    "softwaretesting",
    "solar",
    "tanks",
    "Trucks",
    "virtualization",
    "VPN",
    "VOIP",
    "wifi",
    "windenergy"
]
os.makedirs(DATA_DIR, exist_ok=True)


async def fetch_subreddit_data(reddit, subreddit_name, semaphore):
    async with semaphore:
        dataset = []

        print(f"Collecting data for r/{subreddit_name}.")

        try:
            subreddit = await reddit.subreddit(subreddit_name)

            async for submission in subreddit.new(limit=POST_LIMIT):
                await asyncio.sleep(0.5)
                while True: # avoid to lose post when hitting the API rate
                    try:
                        await submission.load()
                        break
                    except asyncprawcore.exceptions.TooManyRequests:
                        print(f"Rate limite reached on post {submission.id}. Wait for 60 seconds.")
                        await asyncio.sleep(60)
                    except Exception as e:
                        print(f"ERROR to load post {submission.id}, going to next post, {e}")
                        break

                # add post row
                dataset.append({
                    "subreddit": subreddit_name,
                    "post_id": submission.id,
                    "post_title": submission.title,
                    "post_score": submission.score,
                    "post_total_comments": submission.num_comments,
                    "post_body": submission.selftext,
                    "post_url": submission.url,
                    "comment_id": None,
                    "comment_body": None,
                    "comment_score": None,
                    "comment_parent_id": None,
                    "timestamp": submission.created_utc,
                    "content_type": "post"
                })

                try:
                    await submission.comments.replace_more(limit=NESTED_COMMENT_LIMIT)
                except asyncprawcore.exceptions.TooManyRequests:
                    print(f"Rate limite reached on comments for post {submission.id}. Wait for 60 seconds.")
                    await asyncio.sleep(60)
                except Exception as e:
                    print(f"ERROR to extend post {submission.id} ({e})")

                comments = submission.comments.list()[:COMMENTS_LIMIT]

                for comment in comments:
                    # add comment row
                    dataset.append({
                        "subreddit": subreddit_name,
                        "post_id": submission.id,
                        "post_title": submission.title,
                        "post_score": submission.score,
                        "post_total_comments": submission.num_comments,
                        "post_body": submission.selftext,
                        "post_url": submission.url,
                        "comment_id": comment.id,
                        "comment_body": comment.body,
                        "comment_score": comment.score,
                        "comment_parent_id": str(comment.parent_id),
                        "timestamp": comment.created_utc,
                        "content_type": "comment"
                    })

        except asyncprawcore.exceptions.TooManyRequests:
            print(f"Rate limite reached on r/{subreddit_name}. Wait for 60 seconds.")
            await asyncio.sleep(60)
        except Exception as e:
            print(f"Error in r/{subreddit_name}: {e}")

        return dataset


async def collect_all_subreddits():

    reddit = asyncpraw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
    )

    print(f"Starting data collections for all subreddits.")
    start_time = time.time()

    tasks = []
    for subreddit in SUBREDDITS:
        tasks.append(fetch_subreddit_data(reddit, subreddit, semaphore))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed_time = time.time() - start_time
    print(f"Data collection took {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes).")

    aggregated_data = []

    for subreddit, result in zip(SUBREDDITS, results):
        if isinstance(result, Exception):
            print(f"Error for r/{subreddit}: {result}")
        elif result:
            aggregated_data.extend(result)
        else:
            print(f"No data collected for r/{subreddit}.")

    if aggregated_data:
        print(f"Total collected data: {len(aggregated_data)}")
        df = pd.DataFrame(aggregated_data)

        df["row_id"] = range(len(df))

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df["week"] = df["datetime"].dt.strftime("%Y-%U")

        try:
            print(f"Dataset saved to CSV: {OUTPUT_FILE_PATH_CSV}")
            df.to_csv(OUTPUT_FILE_PATH_CSV, index=False)
        except Exception as e:
            print(f"Failed to save CSV: {e}")

        try:
            print(f"Dataset saved to Parquet: {OUTPUT_FILE_PATH_PARQUET}")
            df.to_parquet(OUTPUT_FILE_PATH_PARQUET, index=False)
        except Exception as e:
            print(f"Failed to save Parquet: {e}")
    else:
        print("No data collected.")

    await reddit.close()
    print("All data collected.")

asyncio.run(collect_all_subreddits())