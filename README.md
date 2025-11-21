# BookMate
A lightweight Flask application that provides content-based book recommendations using TF–IDF and cosine similarity, supports mood-based boosts, CSV import, and user feedback collection.

Features

Content-based recommendations using TfidfVectorizer and cosine similarity.

Mood-aware recommendations (boosts based on mood → tag mappings).

Accepts user-provided favourite titles as seeds.

Series continuation suggestions (if series and series_index are present).

CSV upload endpoint to temporarily import/augment book data for the session.

Simple JSON feedback collection saved to feedback.json.
