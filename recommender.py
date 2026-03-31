import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Load the data ──────────────────────────────────────────────
users = pd.read_csv('BX-Users.csv', sep=';',
                    encoding='latin-1', on_bad_lines='skip')
books = pd.read_csv('BX-Books.csv', sep=';',
                    encoding='latin-1', on_bad_lines='skip')
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';',
                      encoding='latin-1', on_bad_lines='skip')

# ── Basic info ─────────────────────────────────────────────────
print("=== USERS ===")
print(f"Rows: {users.shape[0]}, Columns: {users.shape[1]}")
print(users.head(3))

print("\n=== BOOKS ===")
print(f"Rows: {books.shape[0]}, Columns: {books.shape[1]}")
print(books.head(3))

print("\n=== RATINGS ===")
print(f"Rows: {ratings.shape[0]}, Columns: {ratings.shape[1]}")
print(ratings.head(3))
# ── Data Cleaning ──────────────────────────────────────────────

# Keep only explicit ratings (1-10), remove implicit (0)
ratings = ratings[ratings['Book-Rating'] > 0]
print(f"Ratings after removing zeros: {ratings.shape[0]}")

# Keep only users who have rated at least 20 books
user_counts = ratings['User-ID'].value_counts()
ratings = ratings[ratings['User-ID'].isin(
    user_counts[user_counts >= 20].index)]
print(f"Ratings after filtering inactive users: {ratings.shape[0]}")

# Keep only books that have been rated at least 20 times
book_counts = ratings['ISBN'].value_counts()
ratings = ratings[ratings['ISBN'].isin(book_counts[book_counts >= 20].index)]
print(f"Ratings after filtering unpopular books: {ratings.shape[0]}")

print(f"\nUnique users: {ratings['User-ID'].nunique()}")
print(f"Unique books: {ratings['ISBN'].nunique()}")
# ── Build User-Item Matrix ─────────────────────────────────────
user_item_matrix = ratings.pivot_table(
    index='User-ID',
    columns='ISBN',
    values='Book-Rating'
).fillna(0)

print(f"User-Item Matrix Shape: {user_item_matrix.shape}")
print("Sample of matrix:")
print(user_item_matrix.iloc[:5, :5])

# ── Compute User-User Similarity ───────────────────────────────
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

print("User similarity matrix shape:", user_similarity_df.shape)
print("Sample similarities for first user:")
print(user_similarity_df.iloc[0].sort_values(ascending=False).head(5))
# ── Recommendation Function ────────────────────────────────────


def recommend_books(user_id, n_recommendations=5):

    # Check if user exists
    if user_id not in user_item_matrix.index:
        print(f"User {user_id} not found!")
        return

    # Get the top 10 most similar users (excluding the user themselves)
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[
        1:11]

    # Get books the target user has already read
    books_read = set(
        user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)

    # Collect book scores from similar users
    book_scores = {}
    for similar_user, similarity in similar_users.items():
        # Get books this similar user has rated
        similar_user_ratings = user_item_matrix.loc[similar_user]
        rated_books = similar_user_ratings[similar_user_ratings > 0]

        for isbn, rating in rated_books.items():
            if isbn not in books_read:  # Only recommend unread books
                if isbn not in book_scores:
                    book_scores[isbn] = 0
                book_scores[isbn] += similarity * rating

    # Sort by score and get top N
    top_books = sorted(book_scores.items(), key=lambda x: x[1], reverse=True)[
        :n_recommendations]

    # Get book titles
    print(f"\nTop {n_recommendations} recommendations for User {user_id}:")
    print("-" * 50)
    for isbn, score in top_books:
        title_row = books[books['ISBN'] == isbn]
        if not title_row.empty:
            title = title_row.iloc[0]['Book-Title']
            author = title_row.iloc[0]['Book-Author']
            print(f"📚 {title} by {author}")


# ── Test it ────────────────────────────────────────────────────
# Get a real user ID from our dataset
sample_user = user_item_matrix.index[0]
recommend_books(sample_user)

# ── Evaluation ─────────────────────────────────────────────────

# Split ratings into train and test sets
train_data, test_data = train_test_split(
    ratings, test_size=0.2, random_state=42)

print(f"Training set: {train_data.shape[0]} ratings")
print(f"Test set: {test_data.shape[0]} ratings")

# Build matrix from training data only
train_matrix = train_data.pivot_table(
    index='User-ID',
    columns='ISBN',
    values='Book-Rating'
).fillna(0)

# Recompute similarity on training data
train_similarity = cosine_similarity(train_matrix)
train_similarity_df = pd.DataFrame(
    train_similarity,
    index=train_matrix.index,
    columns=train_matrix.index
)

# ── Predict ratings ────────────────────────────────────────────


def predict_rating(user_id, isbn):
    if user_id not in train_similarity_df.index:
        return 0
    if isbn not in train_matrix.columns:
        return 0

    # Get similar users who rated this book
    similar_users = train_similarity_df[user_id].sort_values(ascending=False)[
        1:11]

    numerator = 0
    denominator = 0

    for similar_user, similarity in similar_users.items():
        if similar_user in train_matrix.index:
            rating = train_matrix.loc[similar_user,
                                      isbn] if isbn in train_matrix.columns else 0
            if rating > 0:
                numerator += similarity * rating
                denominator += abs(similarity)

    if denominator == 0:
        return 0
    return numerator / denominator


# ── Compute RMSE on test set ───────────────────────────────────
print("\nEvaluating... (this may take a moment)")

actuals = []
predictions = []

# Test on a sample of 200 to keep it fast
test_sample = test_data.sample(200, random_state=42)

for _, row in test_sample.iterrows():
    predicted = predict_rating(row['User-ID'], row['ISBN'])
    if predicted > 0:
        actuals.append(row['Book-Rating'])
        predictions.append(predicted)

rmse = math.sqrt(mean_squared_error(actuals, predictions))
print(f"RMSE: {rmse:.4f}")
print(f"Evaluated on {len(actuals)} ratings")
print(f"(Lower RMSE = better. On a 1-10 scale, under 2.5 is decent)")
