from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_DIR = os.path.dirname(__file__)
BOOKS_CSV = os.path.join(APP_DIR, 'books_sample.csv')
FEEDBACK_FILE = os.path.join(APP_DIR, 'feedback.json')

app = Flask(__name__)
app.secret_key = 'dev-key'


def load_books(path=BOOKS_CSV):
    df = pd.read_csv(path)
    df.fillna('', inplace=True)
    # create a combined text field for content-based similarity
    df['content'] = (df['title'] + ' ' + df['description'] + ' ' + df['tags']).str.lower()
    return df


def build_vectorizer(contents):
    vect = TfidfVectorizer(max_features=2000, stop_words='english')
    mat = vect.fit_transform(contents)
    return vect, mat


def get_recommendations(df, mat, vect, seed_indices, topn=6, exclude_indices=None, mood_weights=None):
    if exclude_indices is None:
        exclude_indices = set()
    # average seed vectors
    seed_vecs = mat[seed_indices]
    avg = seed_vecs.mean(axis=0)
    sims = cosine_similarity(avg, mat).flatten()
    # apply mood weights by boosting scores of items that mention mood tags
    if mood_weights:
        for idx, row in df.iterrows():
            score_boost = 0.0
            tags = row.get('tags', '')
            for tag, weight in mood_weights.items():
                if tag in tags.lower():
                    score_boost += weight
            sims[idx] = sims[idx] + score_boost

    ranked = sims.argsort()[::-1]
    recs = []
    for i in ranked:
        if i in seed_indices or i in exclude_indices:
            continue
        recs.append(i)
        if len(recs) >= topn:
            break
    return df.iloc[recs]


MOOD_TAG_MAP = {
    'relaxing': {'cozy': 0.3, 'light': 0.2, 'comforting': 0.4},
    'emotional': {'tearjerker': 0.5, 'heartfelt': 0.4},
    'thought-provoking': {'philosophical': 0.5, 'intellectual': 0.4, 'mystery':0.2},
    'adventurous': {'action':0.4, 'adventure':0.5},
    'surprise-me': {}  # special
}


def ensure_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)


@app.route('/')
def index():
    df = load_books()
    vect, mat = build_vectorizer(df['content'])
    # default general recommendations: top popular-ish by cosine to first item
    recs = get_recommendations(df, mat, vect, seed_indices=[0], topn=6)
    return render_template('index.html', recs=recs.to_dict(orient='records'))


@app.route('/recommend', methods=['POST'])
def recommend():
    mood = request.form.get('mood', 'relaxing')
    favourites = request.form.get('favourites', '')
    df = load_books()
    vect, mat = build_vectorizer(df['content'])

    # if user provided favourite titles manually, try to find indices
    seed_indices = []
    if favourites.strip():
        favs = [s.strip().lower() for s in favourites.split('\n') if s.strip()]
        for fav in favs:
            matches = df[df['title'].str.lower().str.contains(fav)]
            if not matches.empty:
                seed_indices.append(matches.index[0])
    # fallback: if no favs, use index 0
    if not seed_indices:
        seed_indices = [0]

    mood_weights = None
    if mood and mood in MOOD_TAG_MAP and mood != 'surprise-me':
        mood_weights = MOOD_TAG_MAP[mood]

    recs = get_recommendations(df, mat, vect, seed_indices=seed_indices, topn=6, mood_weights=mood_weights)

    # Series continuation: check if any favourite is part of a series and suggest next
    series_recs = []
    for idx in seed_indices:
        row = df.loc[idx]
        if row['series']:
            same_series = df[df['series'].str.lower() == row['series'].lower()].sort_values('series_index')
            next_book = same_series[same_series['series_index'] > row['series_index']]
            if not next_book.empty:
                series_recs.append(next_book.iloc[0].to_dict())

    return render_template('recommend.html', recs=recs.to_dict(orient='records'), series_recs=series_recs, mood=mood)


@app.route('/import', methods=['GET', 'POST'])
def import_page():
    # allows CSV upload to replace/augment book list
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.lower().endswith('.csv'):
            dest = os.path.join(APP_DIR, 'user_upload.csv')
            file.save(dest)
            flash('Uploaded CSV saved as user_upload.csv — imported temporary dataset for your session.', 'info')
            return redirect(url_for('index'))
        else:
            flash('Please upload a CSV file.', 'warning')
    return render_template('import.html')


@app.route('/feedback', methods=['POST'])
def feedback():
    ensure_feedback()
    data = request.get_json() or {}
    entry = {
        'title': data.get('title'),
        'liked_aspects': data.get('liked', ''),
        'disliked_aspects': data.get('disliked', ''),
        'rating': data.get('rating')
    }
    with open(FEEDBACK_FILE, 'r+', encoding='utf-8') as f:
        arr = json.load(f)
        arr.append(entry)
        f.seek(0)
        json.dump(arr, f, indent=2)
    return {'status': 'ok'}


if __name__ == '__main__':
    app.run(debug=True)
