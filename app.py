import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Shelft", page_icon="📚", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400;1,600&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --bg: #0A0C10;
    --surface: #12151C;
    --surface2: #1A1E28;
    --border: rgba(255,255,255,0.07);
    --amber: #E8A838;
    --amber-dim: rgba(232,168,56,0.12);
    --text: #F0EDE6;
    --text-muted: #7A7D8A;
    --text-dim: #4A4D5A;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
html, body, [class*="css"], .stApp { background: var(--bg) !important; color: var(--text); }
.block-container { max-width: 1100px !important; padding: 0 2rem 4rem !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── NAVBAR ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 28px 0 28px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 64px;
}
.nav-logo {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.5px;
}
.nav-logo span { color: var(--amber); font-style: italic; }
.nav-pill {
    background: var(--amber-dim);
    border: 1px solid rgba(232,168,56,0.25);
    color: var(--amber);
    padding: 6px 14px;
    border-radius: 100px;
    font-family: 'Outfit', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ── HERO ── */
.hero { margin-bottom: 72px; }
.hero-kicker {
    font-family: 'Outfit', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.hero-kicker::before {
    content: '';
    display: inline-block;
    width: 24px;
    height: 1px;
    background: var(--amber);
}
.hero-headline {
    font-family: 'Playfair Display', serif;
    font-size: clamp(48px, 6vw, 80px);
    font-weight: 400;
    line-height: 1.05;
    color: var(--text);
    margin-bottom: 24px;
    max-width: 720px;
}
.hero-headline em {
    font-style: italic;
    color: var(--amber);
}
.hero-sub {
    font-family: 'Outfit', sans-serif;
    font-size: 16px;
    color: var(--text-muted);
    line-height: 1.7;
    max-width: 480px;
    font-weight: 300;
}

/* ── STATS BAR ── */
.stats-bar {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 56px;
    background: var(--surface);
}
.stat-cell {
    padding: 28px 32px;
    border-right: 1px solid var(--border);
    position: relative;
}
.stat-cell:last-child { border-right: none; }
.stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 36px;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
    margin-bottom: 6px;
}
.stat-val span { color: var(--amber); }
.stat-key {
    font-family: 'Outfit', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-dim);
}

/* ── SEARCH PANEL ── */
.search-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 56px;
    position: relative;
    overflow: hidden;
}
.search-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--amber), transparent);
}
.search-label {
    font-family: 'Outfit', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 24px;
}
.sample-strip {
    font-family: 'Outfit', sans-serif;
    font-size: 12px;
    color: var(--text-dim);
    margin-bottom: 28px;
    background: var(--surface2);
    padding: 10px 16px;
    border-radius: 8px;
    border: 1px solid var(--border);
}
.sample-strip strong { color: var(--text-muted); margin-right: 8px; }

/* ── INPUT OVERRIDES ── */
.stNumberInput input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
}
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stSlider > div { padding: 0 !important; }
label { color: var(--text-muted) !important; font-family: 'Outfit', sans-serif !important; font-size: 12px !important; }

/* ── BUTTON ── */
.stButton > button {
    background: var(--amber) !important;
    color: #0A0C10 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

/* ── SECTION HEADER ── */
.sec-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
}
.sec-title {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 400;
    color: var(--text);
}
.sec-title em { font-style: italic; color: var(--amber); }
.sec-count {
    font-family: 'Outfit', sans-serif;
    font-size: 12px;
    color: var(--text-dim);
    font-weight: 500;
}

/* ── BOOK ROW ── */
.book-row {
    display: flex;
    gap: 20px;
    padding: 20px 0;
    border-bottom: 1px solid var(--border);
    align-items: center;
    transition: background 0.15s;
}
.book-row:last-child { border-bottom: none; }
.book-row:hover { background: rgba(255,255,255,0.02); border-radius: 12px; padding-left: 12px; }
.book-num {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    color: var(--text-dim);
    min-width: 32px;
    text-align: right;
    font-style: italic;
}
.book-cover-wrap { flex-shrink: 0; }
.book-details { flex: 1; min-width: 0; }
.book-name {
    font-family: 'Playfair Display', serif;
    font-size: 17px;
    color: var(--text);
    line-height: 1.3;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.book-by {
    font-family: 'Outfit', sans-serif;
    font-size: 13px;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.book-chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chip {
    font-family: 'Outfit', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 100px;
    border: 1px solid var(--border);
    color: var(--text-dim);
}
.chip-amber {
    border-color: rgba(232,168,56,0.3);
    color: var(--amber);
    background: var(--amber-dim);
}

/* ── METHOD DIVIDER ── */
.method-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 20px;
    font-family: 'Outfit', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dim);
}
.method-divider::before, .method-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── ERROR ── */
.err-box {
    background: rgba(220,53,69,0.08);
    border: 1px solid rgba(220,53,69,0.2);
    border-radius: 12px;
    padding: 20px 24px;
    font-family: 'Outfit', sans-serif;
    font-size: 14px;
    color: #ff6b6b;
}
</style>
""", unsafe_allow_html=True)


# ── DATA ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    books = pd.read_csv('BX-Books.csv', sep=';',
                        encoding='latin-1', on_bad_lines='skip')
    ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';',
                          encoding='latin-1', on_bad_lines='skip')

    ratings = ratings[ratings['Book-Rating'] > 0]
    uc = ratings['User-ID'].value_counts()
    ratings = ratings[ratings['User-ID'].isin(uc[uc >= 20].index)]
    bc = ratings['ISBN'].value_counts()
    ratings = ratings[ratings['ISBN'].isin(bc[bc >= 20].index)]

    mat = ratings.pivot_table(
        index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
    u_sim = pd.DataFrame(cosine_similarity(
        mat), index=mat.index, columns=mat.index)
    i_sim = pd.DataFrame(cosine_similarity(
        mat.T), index=mat.columns, columns=mat.columns)
    return books, ratings, mat, u_sim, i_sim


def enrich(top, books):
    out = []
    for isbn, score in top:
        r = books[books['ISBN'] == isbn]
        if not r.empty:
            out.append({
                'title':  r.iloc[0]['Book-Title'],
                'author': r.iloc[0]['Book-Author'],
                'year':   str(r.iloc[0]['Year-Of-Publication']),
                'cover':  r.iloc[0]['Image-URL-M'],
                'score':  round(score, 2)
            })
    return out


def user_recs(uid, mat, u_sim, books, n):
    if uid not in mat.index:
        return None
    sim_users = u_sim[uid].sort_values(ascending=False)[1:11]
    read = set(mat.loc[uid][mat.loc[uid] > 0].index)
    scores = {}
    for u, s in sim_users.items():
        for isbn, r in mat.loc[u][mat.loc[u] > 0].items():
            if isbn not in read:
                scores[isbn] = scores.get(isbn, 0) + s * r
    return enrich(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n], books)


def item_recs(uid, mat, i_sim, books, n):
    if uid not in mat.index:
        return None
    rated = mat.loc[uid][mat.loc[uid] > 0]
    read = set(rated.index)
    scores = {}
    for isbn, r in rated.items():
        if isbn not in i_sim.index:
            continue
        for s_isbn, s in i_sim[isbn].sort_values(ascending=False)[1:6].items():
            if s_isbn not in read:
                scores[s_isbn] = scores.get(s_isbn, 0) + s * r
    return enrich(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n], books)


def history(uid, mat, books, n=5):
    if uid not in mat.index:
        return None
    rated = mat.loc[uid][mat.loc[uid] > 0].sort_values(ascending=False).head(n)
    out = []
    for isbn, rating in rated.items():
        r = books[books['ISBN'] == isbn]
        if not r.empty:
            out.append({
                'title':  r.iloc[0]['Book-Title'],
                'author': r.iloc[0]['Book-Author'],
                'year':   str(r.iloc[0]['Year-Of-Publication']),
                'cover':  r.iloc[0]['Image-URL-M'],
                'rating': int(rating)
            })
    return out


def book_row(b, idx=None, rating=None):
    c1, c2, c3 = st.columns([0.4, 0.7, 6])
    with c1:
        if idx:
            st.markdown(
                f'<div class="book-num">{idx}</div>', unsafe_allow_html=True)
    with c2:
        try:
            st.image(b['cover'], width=52)
        except:
            st.markdown("📚")
    with c3:
        chips = f'<span class="chip">{b["year"]}</span>'
        if rating:
            chips += f'<span class="chip chip-amber">★ {rating}/10</span>'
        st.markdown(f"""
        <div class="book-row">
            <div class="book-details">
                <div class="book-name">{b['title']}</div>
                <div class="book-by">{b['author']}</div>
                <div class="book-chips">{chips}</div>
            </div>
        </div>""", unsafe_allow_html=True)


# ── UI ─────────────────────────────────────────────────────────
with st.spinner(""):
    books, ratings, mat, u_sim, i_sim = load_data()

# Navbar
st.markdown("""
<div class="navbar">
    <div class="nav-logo">Shel<span>ft</span></div>
    <div class="nav-pill">Book Discovery</div>
</div>""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-kicker">Collaborative Filtering Engine</div>
    <div class="hero-headline">Find your next<br><em>great read.</em></div>
    <div class="hero-sub">Shelft learns from the reading patterns of thousands of real users to surface books you'll actually love — not just what's trending.</div>
</div>""", unsafe_allow_html=True)

# Stats
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-cell">
        <div class="stat-val">{ratings.shape[0]:,}</div>
        <div class="stat-key">Ratings</div>
    </div>
    <div class="stat-cell">
        <div class="stat-val">{mat.shape[0]:,}</div>
        <div class="stat-key">Active Users</div>
    </div>
    <div class="stat-cell">
        <div class="stat-val">{mat.shape[1]:,}</div>
        <div class="stat-key">Books</div>
    </div>
    <div class="stat-cell">
        <div class="stat-val"><span>2.07</span></div>
        <div class="stat-key">RMSE Score</div>
    </div>
</div>""", unsafe_allow_html=True)

# Search panel
st.markdown('<div class="search-panel">', unsafe_allow_html=True)
st.markdown('<div class="search-label">Discover</div>', unsafe_allow_html=True)
sample_ids = ", ".join(map(str, list(mat.index[:10])))
st.markdown(
    f'<div class="sample-strip"><strong>Sample IDs</strong>{sample_ids}</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([3, 2, 2])
with col1:
    user_id = st.number_input("User ID", min_value=1, step=1)
with col2:
    n_recs = st.slider("Results", min_value=1, max_value=10, value=5)
with col3:
    method = st.selectbox("Method", ["User-Based", "Item-Based", "Both"])

find = st.button("Discover Books →")
st.markdown('</div>', unsafe_allow_html=True)

# Results
if find:
    uid = int(user_id)

    st.markdown("""
    <div class="sec-header">
        <div class="sec-title">Reading <em>History</em></div>
        <div class="sec-count">Last 5 rated books</div>
    </div>""", unsafe_allow_html=True)

    hist = history(uid, mat, books)
    if hist is None:
        st.markdown(
            '<div class="err-box">User ID not found. Try one of the sample IDs above.</div>', unsafe_allow_html=True)
    else:
        for i, b in enumerate(hist, 1):
            book_row(b, idx=i, rating=b['rating'])

        st.markdown("""
        <div class="sec-header" style="margin-top:56px">
            <div class="sec-title">Your <em>Recommendations</em></div>
            <div class="sec-count">Picked just for you</div>
        </div>""", unsafe_allow_html=True)

        if method in ["User-Based", "Both"]:
            if method == "Both":
                st.markdown(
                    '<div class="method-divider">User-Based Filtering</div>', unsafe_allow_html=True)
            recs = user_recs(uid, mat, u_sim, books, n_recs)
            if recs:
                for i, b in enumerate(recs, 1):
                    book_row(b, idx=i)

        if method in ["Item-Based", "Both"]:
            if method == "Both":
                st.markdown(
                    '<div class="method-divider">Item-Based Filtering</div>', unsafe_allow_html=True)
            recs = item_recs(uid, mat, i_sim, books, n_recs)
            if recs:
                for i, b in enumerate(recs, 1):
                    book_row(b, idx=i)
