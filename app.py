
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import base64
import time
def get_logo_base64():
  try:
    with open("logo.png", "rb") as img_file:
      return base64.b64encode(img_file.read()).decode()
  except Exception:
    return ""

logo_b64 = get_logo_base64()
# For loading screen (bigger)
logo_img_tag_loading = f'<img src="data:image/png;base64,{logo_b64}" alt="BookMate Logo" style="height:56px; width:auto; margin-bottom:10px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.07);">' if logo_b64 else ''
logo_img_tag = f'<img src="data:image/png;base64,{logo_b64}" alt="BookMate Logo" style="height:36px; width:auto; margin-right:6px; border-radius:6px; box-shadow:0 2px 6px rgba(0,0,0,0.07);">' if logo_b64 else ''
# -------------------- Loading Page --------------------
if 'bookmate_loaded' not in st.session_state:
  st.session_state['bookmate_loaded'] = False

if not st.session_state['bookmate_loaded']:
  st.markdown(f'''
  <style>
  .bookmate-loader-container {{
    height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: #fdf6f8;
  }}
  .bookmate-loader-logo {{
    margin-bottom: 24px;
    display: flex; align-items: center; justify-content: center;
  }}
  .bookmate-dots {{
    display: flex; gap: 16px; margin-bottom: 18px;
  }}
  .bookmate-dot {{
    width: 18px; height: 18px; border-radius: 50%; background: #f7a8c4;
    animation: bookmate-bounce 1s infinite alternate;
  }}
  .bookmate-dot:nth-child(2) {{ background: #f1b7c4; animation-delay: 0.2s; }}
  .bookmate-dot:nth-child(3) {{ background: #e29ca9; animation-delay: 0.4s; }}
  @keyframes bookmate-bounce {{
    0% {{ transform: translateY(0); }}
    100% {{ transform: translateY(-18px); }}
  }}
  .bookmate-tagline {{
    font-family: 'Montserrat', 'Segoe UI', sans-serif; font-size: 1.25rem; color: #d72660; font-weight: 500;
    margin-top: 8px;
    letter-spacing: 0.5px;
  }}
  </style>
  <div class="bookmate-loader-container">
    <div class="bookmate-dots">
      <div class="bookmate-dot"></div>
      <div class="bookmate-dot"></div>
      <div class="bookmate-dot"></div>
    </div>
    <div class="bookmate-tagline">BookMate is here</div>
    <div class="bookmate-loader-logo" style="margin-top:14px;">
      {logo_img_tag_loading}
    </div>
  </div>
  ''', unsafe_allow_html=True)
  time.sleep(2.5)
  st.session_state['bookmate_loaded'] = True
  st.rerun()

# -------------------- Base config --------------------
st.set_page_config(page_title="BookMate", layout="wide")

# -------------------- Pastel background + responsive padding (hard overrides) --------------------
st.markdown("""
<style>
/* FORCE a light pastel app background across Streamlit containers */
:root {
  --bookmate-bg: #fdf6f8;
  --bookmate-ink: #5b3a40;
}
html, body, .stApp, [data-testid="stAppViewContainer"], section.main, section.main > div {
  background-color: var(--bookmate-bg) !important;
  color: var(--bookmate-ink) !important;
}

/* Roomy, responsive side padding + comfy line length */
.block-container {
  padding-left: clamp(50px, 6vw, 120px) !important;
  padding-right: clamp(50px, 6vw, 120px) !important;
  padding-top: 0 !important;
  max-width: 1200px !important;
  margin: 0 auto !important;       /* center the whole app */
}

/* Hide Streamlit's default chrome */
header, footer { visibility: hidden; }

/* Make sidebar light too (if used) */
[data-testid="stSidebar"] {
  background: #fff9fa !important;
  color: var(--bookmate-ink) !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Full-bleed header with About button --------------------
# --- Embed logo.png as base64 for inline HTML image ---

st.markdown(f"""
<div style="
  width: 100vw;
  margin-left: calc(-50vw + 50%);
  background: linear-gradient(90deg, #f7a8c4 0%, #fbc5d8 100%);
  padding: 14px 0;                 /* vertical only; side padding inside */
  border-bottom: 2px solid #f4d9dd;
  box-shadow: 0 4px 14px rgba(245,170,190,.35);
">
  <div style="
      width:100%;
      max-width:1200px;             /* match content width */
      margin: 0 auto;
      padding: 0 clamp(50px, 6vw, 120px);  /* same side padding */
      display:flex; align-items:center; justify-content:space-between;
  ">
    <div style="display:flex; align-items:center; gap:10px;">
      {logo_img_tag}
      <span style="font-family:Poppins,Segoe UI,sans-serif; font-weight:700; font-size:26px; color:#5b3a40;">
        BookMate
      </span>
    </div>
    <a href="#about-bookmate" style="
        background:#ffffffcc; color:#5b3a40; text-decoration:none;
        padding:8px 14px; border-radius:10px; font-weight:600; border:2px solid #f7a8c4;
        transition: all 0.2s ease;
    " onmouseover="this.style.background='#f7a8c4';this.style.color='white';"
      onmouseout="this.style.background='#ffffffcc';this.style.color='#5b3a40';">
      About
    </a>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------- Card/UI styles --------------------
CARD_CSS = """
<style>
:root{
  --bg:#fdf6f8;
  --ink:#5e4b4d;
  --muted:#9b8689;
  --accent:#f1b7c4;
  --accent-dark:#e29ca9;
  --card:#fff9fa;
  --shadow: 0 6px 14px rgba(230,180,190,.25);
  --border:#f4d9dd;
}

/* Global font + colors */
html, body, [class*="css"] {
  font-family: 'Poppins','Segoe UI',sans-serif;
  color: var(--ink);
}

/* Inputs: white boxes for contrast */
input, textarea, select, .stTextInput>div>div>input, .stTextArea textarea {
  background:#ffffff !important;
  color:var(--ink) !important;
  border:1px solid var(--border) !important;
  border-radius:10px !important;
}
input:focus, textarea:focus { border:1px solid var(--accent) !important; outline:none !important; }

/* Tabs accent */
.stTabs [data-baseweb="tab-list"] { gap: 2px; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; }
.stTabs [aria-selected="true"] {
  background: var(--accent) !important; color: white !important; border-radius: 8px !important;
}

/* Cards */
.book-card{
  position:relative;
  background:var(--card);
  border:1px solid var(--border);
  border-radius:16px;
  box-shadow:var(--shadow);
  padding:18px 20px 32px 20px; /* extra bottom padding for button */
  display: flex;
  flex-direction: column;
  transition: transform .2s ease, box-shadow .2s ease;
  margin-bottom: 32px; /* more space between cards */
  height: 100%;
}
.book-card:hover{ transform: translateY(-3px); box-shadow:0 10px 22px rgba(235,170,185,.35); }

.badge{
  position:absolute; right:16px; top:14px;
  background:var(--accent); color:#fff;
  padding:6px 12px; border-radius:10px; font-size:.82rem; font-weight:600;
}
.icon{
  width:42px; height:42px; border:2px solid var(--accent); border-radius:50%;
  display:inline-flex; align-items:center; justify-content:center;
  font-size:20px; color:var(--accent-dark); background:#fdeef2; margin-right:10px;
}
.title{ font-size:1.35rem; line-height:1.25; margin:8px 0 0 0; color:var(--ink); font-weight:600; }
.author{ color:var(--muted); font-style:italic; margin:10px 0 6px 0; }
.dots{ border:none; border-top:2px dotted var(--border); margin:6px 0 10px 0; }
.summary{ color:#6e5c5f; font-size:.98rem; margin:4px 0 0 0; flex: 1 1 auto; }

.find-btn{
  display:inline-block; margin-top:auto;
  background:var(--accent); color:#fff; text-decoration:none;
  padding:10px 16px; border-radius:10px; font-weight:600; font-size:.94rem;
  box-shadow:0 3px 8px rgba(230,150,165,.3);
  margin-bottom: 8px; /* space below button */
  align-self: flex-end;
}
.find-btn:hover{ background:var(--accent-dark); text-decoration:none; }
</style>
"""
st.markdown("""
<style>
/* Center the tab labels and add spacing, and evenly/balanced vertical spacing between lines */
.stTabs [data-baseweb="tab-list"] {
  justify-content: center !important;     /* center the whole tab row */
  margin-top: 18px !important;
  margin-bottom: 18px !important;
}

.stTabs [data-baseweb="tab"] {
  margin: 0 12px !important;              /* space between "Personalized" and "Mood-based" */
  padding: 10px 16px !important;          /* a little more touch target + breathing room */
  border-radius: 12px !important;
  white-space: nowrap;                     /* keep labels on one line */
}
</style>
""", unsafe_allow_html=True)

st.markdown(CARD_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
/* Add vertical space between Streamlit columns (rows of cards) and make columns stretch */
.stColumns > div {
  margin-bottom: 36px !important;
  display: flex;
  flex-direction: column;
  height: 100%;
}
</style>
""", unsafe_allow_html=True)

# -------------------- About section --------------------
st.markdown('<a id="about-bookmate"></a>', unsafe_allow_html=True)
with st.container():
    st.markdown(
        """
<div style="
  background:#ffffff;
  border:2px solid #f7a8c4;
  border-radius:16px;
  padding:18px 20px;
  box-shadow:0 4px 14px rgba(245,170,190,.15);
  margin: 20px 0 10px 0;
">
  <h3 style="margin:0 0 8px 0; color:#5b3a40;">About BookMate</h3>
  <p style="margin:0; color:#5e4b4d;">
  Ever feel like you can't ever seem to finish a book you pick up? Or even pick the right book? Introducing <b>BookMate</b>:
    because finding the right book shouldn’t depend only on genres, ratings, or popularity.
    Traditional systems often miss how a book <i>feels</i>. Many readers get overwhelmed by choice or slip into a reading slump.
    BookMate bridges that gap with a more human, mood-aware approach — considering themes, pacing, tone, emotional feel,
    your moods and your feedback — so your next read feels as personal as a friend’s recommendation.
  </p>
</div>
        """,
        unsafe_allow_html=True
    )

# ------------------------- Data loading -------------------------
@st.cache_data
def load_books(csv_path="books.csv"):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame([
            {"title":"The Brief Wondrous Life of Oscar Wao","author":"Junot Díaz",
             "genres":"Literary Fiction, Historical","tags":"family,identity,character-driven",
             "summary":"A sweeping tale of love, family, and identity set in the Dominican Republic."},
            {"title":"The Stranger","author":"Albert Camus",
             "genres":"Existential Fiction","tags":"philosophical,classic,thought-provoking",
             "summary":"A classic exploration of alienation, morality, and the human condition."},
            {"title":"The Song of Achilles","author":"Madeline Miller",
             "genres":"Historical Fiction, Romance","tags":"mythology,romance,emotional",
             "summary":"A beautifully written reimagining of the Iliad from Patroclus' eyes."},
            {"title":"Quiet Rooms","author":"C. Thinker",
             "genres":"Philosophical, Contemporary","tags":"contemplative,slow-paced,character-driven",
             "summary":"A reflective novel about choices, love, and learning to be with ourselves."},
            {"title":"Skyward Adventures","author":"B. Traveler",
             "genres":"Adventure, Fantasy","tags":"fast-paced,plot-driven,adventurous",
             "summary":"An action-packed quest across floating islands with high stakes."},
        ])
    for col in ["title","author","genres","tags","summary"]:
        if col not in df.columns: df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    df.reset_index(drop=True, inplace=True)
    return df

books_df = load_books()

#Term Frequency–Inverse Document Frequency
@st.cache_data
def build_tfidf_matrix(df):
    #Combines all text fields: itll make it one string
    corpus = (df["title"] + " " + df["author"] + " " + df["genres"] + " " + df["tags"] + " " + df["summary"]).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X = vectorizer.fit_transform(corpus)

    #Learns the vocabulary (unique words) from the corpus.
#Transforms each book into a TF-IDF vector — a sparse matrix of shape (num_books, 2000).'''
    return vectorizer, X

vectorizer, tfidf_matrix = build_tfidf_matrix(books_df)

#converts any given text  into the same TF-IDF vector space as  books
def _text_to_vec(texts):
    if isinstance(texts, (list, tuple)): joined = " ".join([t for t in texts if t])
    else: joined = str(texts or "")
    return vectorizer.transform([joined])

#recommenders

def score_personalized(liked_titles, liked_reviews, disliked_titles, preferred_genres, preferred_authors, top_k=8):
    n = len(books_df); scores = np.zeros(n, dtype=float)

    def split_list(s):
        if not s: return []
        if isinstance(s, (list, tuple)): return [x.strip().lower() for x in s if x.strip()]
        return [x.strip().lower() for x in str(s).splitlines() if x.strip()]

    liked_titles_list = split_list(liked_titles)
    disliked_titles_list = split_list(disliked_titles)
    liked_reviews_list = split_list(liked_reviews)
    genre_list = [g.strip().lower() for g in str(preferred_genres).split(",") if g.strip()] if preferred_genres else []
    author_list = [a.strip().lower() for a in str(preferred_authors).split(",") if a.strip()] if preferred_authors else []

    user_text_parts = liked_titles_list + liked_reviews_list + genre_list + author_list
    if user_text_parts:
        user_vec = _text_to_vec(user_text_parts)
        sim = cosine_similarity(user_vec, tfidf_matrix).flatten()
        scores += sim * 4.0

    for i, row in books_df.iterrows():
        book_genres = [g.strip().lower() for g in row["genres"].split(",") if g.strip()]
        book_author = row["author"].lower()
        genre_overlap = len(set(book_genres) & set(genre_list))
        scores[i] += 3.0 * genre_overlap
        if any(a in book_author for a in author_list): scores[i] += 5.0

    if disliked_titles_list:
        disliked_vec = _text_to_vec(disliked_titles_list)
        sim_dis = cosine_similarity(disliked_vec, tfidf_matrix).flatten()
        scores -= sim_dis * 3.0

    for i, row in books_df.iterrows():
        if row["title"].strip().lower() in map(str.lower, liked_titles_list): scores[i] += 100.0

    top_idx = np.argsort(-scores)[:top_k]
    return books_df.iloc[top_idx].assign(score=scores[top_idx])

def recommend_mood(selected_moods, custom_mood_text, top_k=8):
    mood_map = {
        "Happy":["uplifting","joyful","feel-good","cozy","comforting","feel-good"],
        "Sad":["cathartic","melancholy","healing","emotional", "poignant", "moving"],
        "Romantic":["romance","love","heartwarming", "cute", "feel-good"],
        "Adventurous":["adventure","quest","journey","thrilling","fast-paced","action"],
        "Contemplative":["thoughtful","philosophical","reflective","slow-paced","character-driven"],
        "Mysterious":["mystery","suspense","thriller","uncertainty", "fast--paced"],
        "Nostalgic":["nostalgic","coming-of-age","memory", "sentimental", "reminiscent"],
        "Energetic":["fast-paced","action","adrenaline", "thrilling", "exciting"]
    }
    keywords = []
    for m in selected_moods: keywords += mood_map.get(m, [])
    if custom_mood_text: keywords += [custom_mood_text]

    if not keywords:
        pop_scores = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
        top_idx = np.argsort(-pop_scores)[:top_k]
        return books_df.iloc[top_idx]

    mood_vec = _text_to_vec(keywords)
    sim = cosine_similarity(mood_vec, tfidf_matrix).flatten()

    lower_tags = books_df["tags"].str.lower().fillna("")
    lower_genres = books_df["genres"].str.lower().fillna("")
    combined = (lower_tags + " " + lower_genres)

    tag_scores = []
    for kw in keywords:
        kwl = str(kw).lower()
        tag_scores.append(combined.str.contains(kwl).astype(int))
    tag_scores = np.vstack(tag_scores).sum(axis=0) if tag_scores else np.zeros(len(books_df), dtype=int)

    final_score = sim * 3.0 + tag_scores * 2.5
    top_idx = np.argsort(-final_score)[:top_k]
    return books_df.iloc[top_idx].assign(score=final_score[top_idx])




st.markdown("""
<p style='text-align:center; color:#d72660; margin-top:10px; font-weight:500;'>
  <i>Because the hardest part of reading shouldn’t be choosing what to read next.</i>
</p>
""", unsafe_allow_html=True)
st.write("---")

tab1, tab2 = st.tabs(["Personalized", "Mood-based"])

def render_cards(df, per_row=3, show_button=True):
    for i in range(0, len(df), per_row):
        cols = st.columns(per_row)
        for j, (_, r) in enumerate(df.iloc[i:i+per_row].iterrows()):
            c = cols[j]
            title = r["title"]; author = r["author"]
            genres = r["genres"]; summary = (r["summary"] or "")[:280]
            primary_badge = (genres.split(",")[0] if isinstance(genres, str) and genres else "").strip()
            search_query = urllib.parse.quote_plus(f"{title} {author} book")
            google_link = f"https://www.google.com/search?q={search_query}"
            card_html = f"""
            <div class="book-card">
              <div class="badge">{primary_badge}</div>
              <div class="icon">📖</div>
              <div class="title">{title}</div>
              <div class="author">{author}</div>
              <hr class="dots"/>
              <div class="summary">{summary}...</div>
              {'<a href="'+google_link+'" target="_blank" class="find-btn">Find This Book</a>' if show_button else ''}
            </div>
            """
            c.markdown(card_html, unsafe_allow_html=True)

with tab1:
    st.header("Personalized recommendations")
    st.write("List a few favorites and what you liked, a few you didn’t, and your preferred genres/authors.")

    col1, col2 = st.columns([2,1])
    with col1:
        liked_books_input = st.text_area("❤️ Favorite books (one per line)", height=120)
        disliked_books_input = st.text_area("✖ Not-so-favorite books (optional)", height=80)
    with col2:
        pref_genres = st.text_input("⭐ Preferred genres (comma separated)")
        pref_authors = st.text_input("👤 Favorite authors (comma separated)")

    liked_reviews_text = st.text_area("📝 What you liked about those books", height=80)

    if st.button("Get Personalized Recommendations"):
        with st.spinner("Finding matches..."):
            recs = score_personalized(liked_books_input, liked_reviews_text, disliked_books_input, pref_genres, pref_authors, top_k=12)
            st.success("Here are some picks for you!")
            render_cards(recs, per_row=3, show_button=True)

with tab2:
    st.header("Mood-based recommendations")
    st.write("Pick one or more moods, or describe your mood in your own words.")

    mood_options = ["Happy","Sad","Romantic","Adventurous","Contemplative","Mysterious","Nostalgic","Energetic"]
    selected = st.multiselect("Pick moods", mood_options, default=[])
    custom_mood = st.text_input("✍️ Custom mood (free text)", "")

    if st.button("Get Mood-Based Recommendations"):
        with st.spinner("Matching your mood..."):
            recs = recommend_mood(selected, custom_mood, top_k=12)
            st.success("Books that fit your vibe:")
            render_cards(recs, per_row=2, show_button=True)
