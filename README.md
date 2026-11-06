# BookMate 

A personalised, mood‑aware and preference‑driven **book recommendation app** built entirely in **Streamlit**, using TF–IDF similarity, genre/author boosts, mood mapping, and personalized text‑based signals.

**Features**

Personalized Recommendations

* Enter favourite books, disliked ones, reviews, preferred genres, and authors.
* TF–IDF vectors + cosine similarity score how well each book matches your personal taste.
* Genre and author overlap boosts relevance.

Mood‑Based Recommendations

* Choose from predefined moods (Happy, Sad, Romantic, Adventurous, etc.).
* Or type a **custom mood** — the model converts it into a semantic vector.
* Uses both similarity scores and tag/genre keyword matching.

Elegant UI & Animations

* Custom full‑screen loader animation.
* Soft pastel theme, custom CSS, centered tabs, responsive card grid.
* Modern cards: genre badges, summaries, Google search button.
* Clean header with inline logo + About section.

### Data Handling

* Automatically loads `books.csv` if present.
* Falls back to 5 curated sample books.
* Caches TF–IDF and dataset for fast re-runs.

TF–IDF Engine
* Combines *title + author + genres + tags + summary* into a single text corpus.
* Vectorizes it with `TfidfVectorizer(max_features=2000, stop_words='english')`.
* Uses cosine similarity to compare user preferences/moods with each book.

Personalized Model
* Positive similarity boosts.
* Negative similarity penalties for disliked books.
* Extra scoring for genre matches and author matches.

Mood Model
* Maps moods → keyword lists.
* Computes similarity + tag matching for final ranking.



UI Highlights
* Fully custom CSS: cards, badges, buttons, sidebar, tabs.
* Full‑bleed gradient header with About button.
* Animated loading screen with bouncing dots.
* Responsive layout with adjustable columns.


Acknowledgements

Built with Streamlit, pandas, NumPy, scikit‑learn, and lots of pink pastel love.
