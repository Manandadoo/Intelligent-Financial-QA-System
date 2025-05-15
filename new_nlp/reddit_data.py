import praw
import pandas as pd

# Create a Reddit instance
reddit = praw.Reddit(
    client_id="TgOyExrtDOKt6rmmoYThGw",
    client_secret="6B6gmTJ2fXNXe5NnCz9od7MFMc98Yw",
    user_agent="Anirudh"
)

# Fetch finance-related posts about Apple
subreddit = reddit.subreddit("all")
query = "Apple finance OR stock OR investment OR earnings"
posts = []

for submission in subreddit.search(query, limit=50):
    posts.append([submission.title, submission.score, submission.subreddit, submission.url, submission.created_utc, submission.selftext])

# Save to DataFrame
# Save to DataFrame
df = pd.DataFrame(posts, columns=["Title", "Score", "Subreddit", "URL", "Date", "Content"])

# Export to CSV in your home directory
df.to_csv("/Users/anirudhgupta/Desktop/apple_finance_reddit.csv", index=False, encoding="utf-8")
print("Data saved to apple_finance_reddit.csv on Desktop")

