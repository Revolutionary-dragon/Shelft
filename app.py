import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="Shelft", page_icon="📚", layout="wide")

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .book-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #e94560;
    }
    .book-title { font-size: 16px; font-weight: bold; color: #ffffff; }
    .book-author { font-size: 13px; color: #a0a0a0; }
    .book-year { font-size: 12px; color: #e94560; }
    .section-header {
        font-size: 22px;
        font-weight: bold;
        color: #e94560;
        margin: 20px 0 10px 0;
    }
    .stat-box {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #e94560;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────


@st.cache_data
def load_data():
    users = pd.read_csv('BX-Users.csv', sep=';',
                        encoding='latin-1', on_bad_lines='skip')
    books = pd.read_csv('BX-Books.csv', sep=';',
                        encoding='latin-1', on_bad_lines='skip')
    ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';',
                          encoding='latin-1', on_bad_lines='skip')

    ratings = ratings[ratings['Book-Rating'] > 0]
    user_counts = ratings['User-ID'].value_counts()
    ratings = ratings[ratings['User-ID'].isin(
        user_counts[user_counts >= 20].index)]
    book_counts = ratings['ISBN'].value_counts()
    ratings = ratings[ratings['ISBN'].isin(
        book_counts[book_counts >= 20].index)]

    user_item_matrix = ratings.pivot_table(
        index='User-ID', columns='ISBN', values='Book-Rating'
    ).fillna(0)

    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    return books, ratings, user_item_matrix, user_similarity_df, item_similarity_df

# ── Recommend functions ────────────────────────────────────────


def recommend_user_based(user_id, user_item_matrix, user_similarity_df, books, n=5):
    if user_id not in user_item_matrix.index:
        return None
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[
        1:11]
    books_read = set(
        user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
    book_scores = {}
    for similar_user, similarity in similar_users.items():
        rated = user_item_matrix.loc[similar_user]
        rated = rated[rated > 0]
        for isbn, rating in rated.items():
            if isbn not in books_read:
                book_scores[isbn] = book_scores.get(
                    isbn, 0) + similarity * rating
    top_books = sorted(book_scores.items(),
                       key=lambda x: x[1], reverse=True)[:n]
    return enrich_books(top_books, books)


def recommend_item_based(user_id, user_item_matrix, item_similarity_df, books, n=5):
    if user_id not in user_item_matrix.index:
        return None
    user_ratings = user_item_matrix.loc[user_id]
    rated_books = user_ratings[user_ratings > 0]
    books_read = set(rated_books.index)
    book_scores = {}
    for isbn, rating in rated_books.items():
        if isbn not in item_similarity_df.index:
            continue
        similar_books = item_similarity_df[isbn].sort_values(ascending=False)[
            1:6]
        for similar_isbn, similarity in similar_books.items():
            if similar_isbn not in books_read:
                book_scores[similar_isbn] = book_scores.get(
                    similar_isbn, 0) + similarity * rating
    top_books = sorted(book_scores.items(),
                       key=lambda x: x[1], reverse=True)[:n]
    return enrich_books(top_books, books)


def enrich_books(top_books, books):
    results = []
    for isbn, score in top_books:
        row = books[books['ISBN'] == isbn]
        if not row.empty:
            results.append({
                'Title': row.iloc[0]['Book-Title'],
                'Author': row.iloc[0]['Book-Author'],
                'Year': row.iloc[0]['Year-Of-Publication'],
                'Cover': row.iloc[0]['Image-URL-M'],
                'Score': round(score, 2)
            })
    return results


def get_reading_history(user_id, user_item_matrix, books, n=5):
    if user_id not in user_item_matrix.index:
        return None
    user_ratings = user_item_matrix.loc[user_id]
    rated = user_ratings[user_ratings > 0].sort_values(ascending=False).head(n)
    results = []
    for isbn, rating in rated.items():
        row = books[books['ISBN'] == isbn]
        if not row.empty:
            results.append({
                'Title': row.iloc[0]['Book-Title'],
                'Author': row.iloc[0]['Book-Author'],
                'Year': row.iloc[0]['Year-Of-Publication'],
                'Cover': row.iloc[0]['Image-URL-M'],
                'Rating': int(rating)
            })
    return results


def render_book_card(book, show_rating=False, rating=None):
    col1, col2 = st.columns([1, 4])
    with col1:
        try:
            st.image(book['Cover'], width=80)
        except:
            st.markdown("📚")
    with col2:
        st.markdown(
            f"<div class='book-title'>{book['Title']}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='book-author'>by {book['Author']}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='book-year'>📅 {book['Year']}</div>", unsafe_allow_html=True)
        if show_rating and rating:
            st.markdown(f"⭐ Your rating: **{rating}/10**")
    st.divider()


# ── Main UI ────────────────────────────────────────────────────
st.markdown("<h1 style='color:#e94560'>📚 Shelft</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#a0a0a0; font-size:16px; margin-top:-15px;'>Your personal book recommendation system</p>", unsafe_allow_html=True)

with st.spinner("Loading data and computing similarities..."):
    books, ratings, user_item_matrix, user_similarity_df, item_similarity_df = load_data()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"<div class='stat-box'><h3>{ratings.shape[0]:,}</h3><p>Ratings</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown(
        f"<div class='stat-box'><h3>{user_item_matrix.shape[0]:,}</h3><p>Users</p></div>", unsafe_allow_html=True)
with col3:
    st.markdown(
        f"<div class='stat-box'><h3>{user_item_matrix.shape[1]:,}</h3><p>Books</p></div>", unsafe_allow_html=True)

st.markdown("---")

st.markdown("**Sample User IDs:**")
st.code(", ".join(map(str, list(user_item_matrix.index[:10]))))

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
with col2:
    n_recs = st.slider("Recommendations:", min_value=1, max_value=10, value=5)
with col3:
    method = st.selectbox("Method:", ["User-Based", "Item-Based", "Both"])

if st.button("Get Recommendations 📚", use_container_width=True):
    uid = int(user_id)

    st.markdown("<div class='section-header'>📖 Reading History</div>",
                unsafe_allow_html=True)
    history = get_reading_history(uid, user_item_matrix, books, n=5)
    if history is None:
        st.error("User ID not found. Try one of the sample IDs above.")
    else:
        for item in history:
            render_book_card(item, show_rating=True, rating=item['Rating'])

        st.markdown(
            "<div class='section-header'>🎯 Recommendations</div>", unsafe_allow_html=True)

        if method in ["User-Based", "Both"]:
            if method == "Both":
                st.markdown("##### 👥 User-Based")
            recs = recommend_user_based(
                uid, user_item_matrix, user_similarity_df, books, n_recs)
            if recs:
                for book in recs:
                    render_book_card(book)

        if method in ["Item-Based", "Both"]:
            if method == "Both":
                st.markdown("##### 📘 Item-Based")
            recs = recommend_item_based(
                uid, user_item_matrix, item_similarity_df, books, n_recs)
            if recs:
                for book in recs:
                    render_book_card(book)
