"""CineSearch Flask application module.

This module implements a service-oriented Flask web application for the
"An Open-Source Movie Browsing and Discovery Platform" project. The
application uses SQLite through Flask-SQLAlchemy to persist user accounts,
favorites, and reviews as locally cached metadata, while the TMDB API is used
to retrieve dynamic movie discovery, search, trending, and metadata payloads
at request time.
"""

import os
import random
from collections import Counter
from datetime import datetime, timedelta
from flask import Flask, render_template, request, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import requests
from dotenv import load_dotenv
from sqlalchemy import inspect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import generate_password_hash, check_password_hash

load_dotenv()

app = Flask(__name__)
app.secret_key = "super_secret_key"
basedir = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "cinesearch.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    favorite_genre = db.Column(db.String(100), nullable=True)
    reviews = db.relationship("Review", back_populates="user", lazy=True, cascade="all, delete-orphan")
    favorites = db.relationship("Favorite", back_populates="user", lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)
    movie_title = db.Column(db.String(255), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    review_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    user = db.relationship("User", back_populates="reviews")


class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(255), nullable=False)
    poster_path = db.Column(db.String(255), nullable=True)
    user = db.relationship("User", back_populates="favorites")

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

GENRES = [
    {"id": 28,    "name": "Action"},
    {"id": 12,    "name": "Adventure"},
    {"id": 16,    "name": "Animation"},
    {"id": 35,    "name": "Comedy"},
    {"id": 80,    "name": "Crime"},
    {"id": 99,    "name": "Documentary"},
    {"id": 18,    "name": "Drama"},
    {"id": 10751, "name": "Family"},
    {"id": 14,    "name": "Fantasy"},
    {"id": 27,    "name": "Horror"},
    {"id": 9648,  "name": "Mystery"},
    {"id": 10749, "name": "Romance"},
    {"id": 878,   "name": "Sci-Fi"},
    {"id": 53,    "name": "Thriller"},
    {"id": 10752, "name": "War"},
    {"id": 37,    "name": "Western"},
]

POPULAR_LABELS = {
    "week":       "This Week",
    "month":      "This Month",
    "all-time":   "All Time",
    "most-voted": "Most Voted",
}


def ensure_review_schema():
    """Ensures the local SQLite review table matches the expected schema."""
    inspector = inspect(db.engine)
    if "review" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("review")}

    if "movie_title" not in existing_columns:
        db.session.execute(db.text("ALTER TABLE review ADD COLUMN movie_title VARCHAR(255)"))

    if "review_text" not in existing_columns:
        db.session.execute(db.text("ALTER TABLE review ADD COLUMN review_text TEXT"))

    if "movie_title" not in existing_columns or "review_text" not in existing_columns:
        review_text_source = "content" if "content" in existing_columns else "''"
        db.session.execute(
            db.text(
                f"""
                UPDATE review
                SET movie_title = COALESCE(movie_title, 'Untitled'),
                    review_text = COALESCE(review_text, {review_text_source}, '')
                """
            )
        )
        # Persist the one-time schema migration so subsequent requests can rely
        # on the normalized review structure stored in SQLite.
        db.session.commit()

def fetch_tmdb_movie_details(movie_id):
    """Retrieves one TMDB movie payload for downstream recommendation tasks."""
    if not API_KEY:
        return None

    try:
        movie_url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}"
        response = requests.get(movie_url, timeout=10)
        return response.json()
    except Exception:
        return None


def select_profile_favorites(favorites, sample_size=5):
    """Selects a compact subset of favorites to build the user profile text."""
    # The favorites table does not store TMDB metadata snapshots such as
    # overview or genre identifiers, so the recommendation route samples a
    # small subset of the most recently saved favorites and enriches them with
    # live TMDB metadata before vectorization.
    ordered_favorites = sorted(favorites, key=lambda favorite: favorite.id, reverse=True)
    return ordered_favorites[:sample_size]


def build_user_preference_profile(profile_movies):
    """Builds a single textual profile and genre histogram from favorite movies."""
    user_profile_segments = []
    genre_counter = Counter()
    genre_names = {}

    for movie in profile_movies:
        title = movie.get("title", "").strip()
        overview = movie.get("overview", "").strip()

        # Content-Based Filtering begins by transforming a user's observed
        # history into text. Concatenating titles and plot overviews creates a
        # compact "User Preference Profile" that captures recurring narrative
        # themes, topics, and descriptive vocabulary from the user's favorites.
        combined_text = f"{title} {overview}".strip()
        if combined_text:
            user_profile_segments.append(combined_text)

        for genre in movie.get("genres", []):
            genre_id = genre.get("id")
            genre_name = genre.get("name")
            if genre_id:
                genre_counter[genre_id] += 1
                if genre_name:
                    genre_names[genre_id] = genre_name

    user_profile_text = " ".join(user_profile_segments).strip()
    return user_profile_text, genre_counter, genre_names


def fetch_genre_candidate_movies(genre_id, max_candidates=40):
    """Fetches a genre-constrained candidate pool from TMDB for re-ranking."""
    if not API_KEY or not genre_id:
        return []

    candidate_movies = []

    try:
        for page in range(1, 3):
            # TMDB discovery provides an efficient first-stage retrieval step:
            # the candidate pool is limited to a dominant genre so TF-IDF
            # ranking is applied only to movies that are already broadly
            # aligned with the user's historic taste profile.
            url = (
                f"{BASE_URL}/discover/movie?api_key={API_KEY}"
                f"&with_genres={genre_id}&sort_by=popularity.desc"
                f"&vote_count.gte=25&page={page}"
            )
            data = requests.get(url, timeout=10).json()
            results = data.get("results", [])
            if not results:
                break
            candidate_movies.extend(results)
            if len(candidate_movies) >= max_candidates:
                break
    except Exception:
        return []

    return candidate_movies[:max_candidates]


def fetch_profile_candidate_movies(profile_movies, genre_counter, max_candidates=80):
    """Builds a richer candidate pool from the user's full favorite profile."""
    if not API_KEY:
        return []

    candidate_movies = {}
    prioritized_genres = [genre_id for genre_id, _ in genre_counter.most_common(3)]

    def store_candidates(movies):
        for movie in movies:
            movie_id = movie.get("id")
            if movie_id and movie_id not in candidate_movies:
                candidate_movies[movie_id] = movie
            if len(candidate_movies) >= max_candidates:
                break

    try:
        # Blend multiple content-derived retrieval strategies so the
        # recommendations page is driven by the user's full viewing profile
        # rather than by one dominant genre alone.
        for movie in profile_movies:
            movie_id = movie.get("id")
            if not movie_id:
                continue

            for endpoint in ("similar", "recommendations"):
                url = f"{BASE_URL}/movie/{movie_id}/{endpoint}?api_key={API_KEY}&page=1"
                data = requests.get(url, timeout=10).json()
                results = data.get("results", [])
                if results:
                    store_candidates(results)
                if len(candidate_movies) >= max_candidates:
                    return list(candidate_movies.values())[:max_candidates]

        for genre_id in prioritized_genres:
            for page in range(1, 3):
                url = (
                    f"{BASE_URL}/discover/movie?api_key={API_KEY}"
                    f"&with_genres={genre_id}&sort_by=popularity.desc"
                    f"&vote_count.gte=25&page={page}"
                )
                data = requests.get(url, timeout=10).json()
                results = data.get("results", [])
                if not results:
                    break
                store_candidates(results)
                if len(candidate_movies) >= max_candidates:
                    return list(candidate_movies.values())[:max_candidates]
    except Exception:
        return list(candidate_movies.values())[:max_candidates]

    return list(candidate_movies.values())[:max_candidates]


def rank_movies_by_user_profile(user_profile_text, candidate_movies, excluded_movie_ids=None, top_k=10):
    """Ranks candidate movies by TF-IDF cosine similarity against user history."""
    excluded_movie_ids = excluded_movie_ids or set()

    candidate_records = []
    candidate_texts = []

    for movie in candidate_movies:
        if movie.get("id") in excluded_movie_ids:
            continue

        title = movie.get("title", "").strip()
        overview = movie.get("overview", "").strip()
        combined_text = f"{title} {overview}".strip()

        if not combined_text:
            continue

        candidate_records.append(movie)
        candidate_texts.append(combined_text)

    if not user_profile_text or not candidate_records:
        return []

    try:
        # Academic note: TF-IDF vectorization converts the user's aggregated
        # viewing-history text and each candidate movie's metadata into
        # weighted term vectors. Cosine similarity is then used to quantify
        # semantic alignment between the user profile and each candidate,
        # thereby implementing true Content-Based Filtering rather than a
        # genre-only heuristic.
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([user_profile_text] + candidate_texts)
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    except Exception:
        return []

    scored_candidates = []
    for similarity_score, movie in zip(similarity_scores, candidate_records):
        movie["similarity_score"] = float(similarity_score)
        scored_candidates.append((similarity_score, movie))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    return [movie for score, movie in scored_candidates[:top_k]]


def get_content_recommendations(target_movie_id, target_overview, genre_id):
    """Builds lightweight content-based recommendations from TMDB metadata."""
    if not API_KEY or not genre_id:
        return []

    candidate_movies = []
    try:
        for page in range(1, 4):
            # The request includes the TMDB API key and discovery filters so the
            # candidate pool stays within the same genre and remains lightweight.
            url = (
                f"{BASE_URL}/discover/movie?api_key={API_KEY}"
                f"&with_genres={genre_id}&page={page}&sort_by=popularity.desc&vote_count.gte=10"
            )
            # Parse the JSON payload returned by TMDB into a Python dictionary
            # for downstream ranking operations.
            data = requests.get(url, timeout=10).json()
            results = data.get("results", [])
            if not results:
                break
            candidate_movies.extend(results)
            if len(candidate_movies) >= 45:
                break
    except Exception:
        return []

    records = []
    texts = []
    for movie in candidate_movies:
        if movie.get("id") == target_movie_id:
            continue
        title = movie.get("title", "")
        overview = movie.get("overview", "")
        combined = f"{title} {overview}".strip()
        if not combined:
            combined = title or ""
        records.append(movie)
        texts.append(combined)

    if not records:
        return []

    target_text = target_overview or ""
    corpus = [target_text] + texts
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    except Exception:
        return []

    scored_movies = list(zip(similarity_scores, records))
    scored_movies.sort(key=lambda item: item[0], reverse=True)

    recommended = [movie for score, movie in scored_movies[:4]]
    return recommended

@app.route("/")
def home():
    """Handles retrieval and display of trending content on the home page.

    Session Authentication:
        Not required.

    Data Sources:
        Communicates with the TMDB API for trending movie data and does not
        query the local SQLite database.
    """
    if not API_KEY:
        return render_template("index.html", movies=[], genres=GENRES, current_page=1, total_pages=1, error="API_KEY not configured. Please check your .env file.")
    
    try:
        page = request.args.get("page", 1, type=int)
        # The API key authenticates the TMDB request, while the page parameter
        # supports server-side pagination for the trending listing.
        url = f"{BASE_URL}/trending/movie/week?api_key={API_KEY}&page={page}"
        response = requests.get(url, timeout=10)
        # TMDB responses are returned as JSON and converted into a dictionary so
        # the template can iterate over the movie metadata payload.
        data = response.json()
        
        if "status_code" in data and data["status_code"] != 200:
            return render_template("index.html", movies=[], genres=GENRES, current_page=1, total_pages=1, error=f"API Error: {data.get('status_message', 'Unknown error')}")
        
        return render_template("index.html", movies=data.get("results", []), genres=GENRES, current_page=page, total_pages=data.get("total_pages", 1))
    except Exception as e:
        return render_template("index.html", movies=[], genres=GENRES, current_page=1, total_pages=1, error=f"Error fetching movies: {str(e)}")

@app.route("/search")
def search():
    """Handles keyword-based movie retrieval for the search interface.

    Session Authentication:
        Not required.

    Data Sources:
        Communicates with the TMDB API search endpoint and does not query the
        local SQLite database.
    """
    if not API_KEY:
        return render_template("index.html", movies=[], genres=GENRES, current_page=1, total_pages=1, error="API_KEY not configured. Please check your .env file.")
    
    try:
        query = request.args.get("q", "")
        page = request.args.get("page", 1, type=int)
        # The query string is forwarded to TMDB together with the API key and
        # pagination parameters to retrieve matching movie records.
        url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={query}&page={page}"
        response = requests.get(url, timeout=10)
        # JSON parsing transforms the remote API payload into a template-ready
        # structure containing the movie list and pagination metadata.
        data = response.json()
        
        if "status_code" in data and data["status_code"] != 200:
            return render_template("index.html", movies=[], genres=GENRES, current_page=1, total_pages=1, error=f"API Error: {data.get('status_message', 'Unknown error')}")
        
        return render_template("index.html", movies=data.get("results", []), genres=GENRES, current_page=page, total_pages=data.get("total_pages", 1))
    except Exception as e:
        return render_template("index.html", movies=[], genres=GENRES, current_page=1, total_pages=1, error=f"Error searching movies: {str(e)}")

@app.route("/browse")
def browse():
    """Handles filtered browsing across genres, ratings, popularity, and years.

    Session Authentication:
        Not required.

    Data Sources:
        Communicates with the TMDB API discover endpoint and does not query the
        local SQLite database.
    """
    page = request.args.get("page", 1, type=int)
    year     = request.args.get("year")
    rating   = request.args.get("rating")
    popular  = request.args.get("popular")
    genre_id = request.args.get("genre")

    params = {"page": page}

    DECADE_MAP = {
        "2020s": ("2020-01-01", "2029-12-31"),
        "2010s": ("2010-01-01", "2019-12-31"),
        "2000s": ("2000-01-01", "2009-12-31"),
        "1990s": ("1990-01-01", "1999-12-31"),
        "1980s": ("1980-01-01", "1989-12-31"),
        "1970s": ("1970-01-01", "1979-12-31"),
    }

    if year and year in DECADE_MAP:
        date_gte, date_lte = DECADE_MAP[year]
        params["primary_release_date.gte"] = date_gte
        params["primary_release_date.lte"] = date_lte
    elif year:
        params["primary_release_year"] = year

    if rating:
        params["vote_average.gte"] = rating
        params["vote_count.gte"]   = 200

    sort_map = {
        "week":       "popularity.desc",
        "month":      "popularity.desc",
        "all-time":   "popularity.desc",
        "most-voted": "vote_count.desc",
    }
    params["sort_by"] = sort_map.get(popular, "popularity.desc")

    if popular and not year:
        today = datetime.now().date()
        if popular == "week":
            params["primary_release_date.gte"] = str(today - timedelta(days=7))
        elif popular == "month":
            params["primary_release_date.gte"] = str(today - timedelta(days=30))

    if genre_id:
        params["with_genres"] = genre_id

    # The discover request is assembled dynamically so only the selected filter
    # parameters are transmitted to TMDB.
    url = f"{BASE_URL}/discover/movie?api_key={API_KEY}"
    for k, v in params.items():
        url += f"&{k}={v}"

    try:
        response = requests.get(url, timeout=10)
        # The JSON body contains the filtered catalog payload and pagination
        # values used by the front-end browse view.
        data = response.json()
        
        if "status_code" in data and data["status_code"] != 200:
            error_msg = f"API Error: {data.get('status_message', 'Unknown error')}"
        else:
            error_msg = None
    except Exception as e:
        data = {"results": [], "total_pages": 1}
        error_msg = f"Error fetching movies: {str(e)}"

    genre_name = next(
        (g["name"] for g in GENRES if str(g["id"]) == str(genre_id)), None
    ) if genre_id else None

    return render_template(
        "index.html",
        movies            = data.get("results", []),
        genres            = GENRES,
        active_year       = year,
        active_rating     = rating,
        active_popular    = popular,
        active_genre      = genre_id,
        active_genre_name = genre_name,
        popular_labels    = POPULAR_LABELS,
        current_page      = page,
        total_pages       = data.get("total_pages", 1),
        error             = error_msg,
    )

@app.route("/movie/<int:movie_id>")
def movie_details(movie_id):
    """Handles retrieval and display of detailed metadata for one movie.

    Session Authentication:
        Not required for viewing details. Session data is optionally consulted
        to determine whether the current movie is already in the user's
        favorites.

    Data Sources:
        Communicates with the TMDB API for movie metadata and credits, and
        queries the local SQLite database for favorites and reviews.
    """
    # The TMDB request appends credits so the route can build a richer detail
    # page without issuing separate network requests for cast and crew data.
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&append_to_response=credits"
    response = requests.get(url).json()
    
    if "id" not in response:
        return render_template("404.html", message="Movie not found"), 404
    
    # Extract director from crew
    director = None
    credits = response.get("credits", {})
    crew = credits.get("crew", [])
    for person in crew:
        if person.get("job") == "Director":
            director = person.get("name")
            break
    
    # Extract top 5 cast members
    cast = credits.get("cast", [])[:5]
    cast_list = [{"name": person.get("name"), "character": person.get("character")} for person in cast]
    
    # Extract genres
    genres = response.get("genres", [])
    genre_names = [g.get("name") for g in genres]
    
    # Extract release year from release_date
    release_date = response.get("release_date", "")
    release_year = release_date.split("-")[0] if release_date else "N/A"
    
    movie_data = {
        "id": response.get("id"),
        "title": response.get("title", "Untitled"),
        "overview": response.get("overview", "No synopsis available"),
        "poster_path": response.get("poster_path"),
        "backdrop_path": response.get("backdrop_path"),
        "vote_average": response.get("vote_average", 0),
        "release_date": release_date,
        "release_year": release_year,
        "genres": genre_names,
        "director": director,
        "cast": cast_list,
        "runtime": response.get("runtime", "N/A"),
        "budget": response.get("budget", 0),
        "revenue": response.get("revenue", 0),
    }

    primary_genre_id = None
    if genres:
        primary_genre_id = genres[0].get("id")

    recommendations = get_content_recommendations(
        target_movie_id=movie_id,
        target_overview=response.get("overview", ""),
        genre_id=primary_genre_id,
    )

    user_id = session.get("user_id")
    is_favorite = False
    if user_id:
        is_favorite = Favorite.query.filter_by(user_id=user_id, movie_id=movie_id).first() is not None

    movie_reviews = Review.query.filter_by(movie_id=movie_id).order_by(Review.created_at.desc()).all()

    return render_template(
        "details.html",
        movie=movie_data,
        recommendations=recommendations,
        is_favorite=is_favorite,
        movie_reviews=movie_reviews,
    )

@app.route("/api/search-suggestions")
def search_suggestions():
    """Provides lightweight JSON search suggestions for the live search box.

    Session Authentication:
        Not required.

    Data Sources:
        Communicates with the TMDB API search endpoint and does not query the
        local SQLite database.
    """
    query = request.args.get("q", "")
    
    if len(query) < 2:
        return {"suggestions": []}
    
    # The API key authorizes the remote query, and the first page is sufficient
    # for compact type-ahead suggestions.
    url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={query}&page=1"
    data = requests.get(url).json()
    movies = data.get("results", [])
    
    # Return full movie data instead of just titles
    suggestions = [
        {
            "title": movie.get("title", "Untitled"),
            "poster_path": movie.get("poster_path"),
            "release_date": movie.get("release_date", "N/A"),
            "vote_average": movie.get("vote_average", 0)
        }
        for movie in movies[:8]
    ]
    return {"suggestions": suggestions}

@app.route("/api/random-movie")
def random_movie():
    """Returns a randomly selected trending movie as a JSON payload.

    Session Authentication:
        Not required.

    Data Sources:
        Communicates with the TMDB API and does not query the local SQLite
        database.
    """
    page = random.randint(1, 50)
    # The random page parameter broadens the sample space before a final movie
    # is selected from the TMDB trending payload.
    url = f"{BASE_URL}/trending/movie/week?api_key={API_KEY}&page={page}"
    data = requests.get(url).json()
    movies = data.get("results", [])
    
    if not movies:
        return {"movie": None}
    
    random_movie = random.choice(movies)
    return {"movie": random_movie}


@app.route("/recommendations", methods=["GET"])
def recommendations():
    """Generates personalized recommendations from favorited movie metadata.

    Session Authentication:
        Required. The route depends on the authenticated user's session to
        identify favorites stored in the local SQLite database.

    Data Sources:
        Queries the local SQLite database for favorites and communicates with
        the TMDB API for favorite-movie enrichment and genre-constrained
        candidate discovery prior to TF-IDF ranking.
    """
    user_id = session.get("user_id")
    if not user_id:
        flash("Please sign in to get personalized recommendations.", "error")
        return redirect(url_for("auth"))

    if not API_KEY:
        flash("Movie recommendations are unavailable because the API key is not configured.", "error")
        return redirect(url_for("home"))

    favorites = Favorite.query.filter_by(user_id=user_id).all()
    if not favorites:
        flash("Add some movies to your Favorites first to get personalized recommendations!", "error")
        return redirect(url_for("home"))

    sampled_favorites = select_profile_favorites(favorites, sample_size=5)
    enriched_profile_movies = []

    try:
        for favorite in sampled_favorites:
            movie_data = fetch_tmdb_movie_details(favorite.movie_id)
            if movie_data and movie_data.get("id"):
                enriched_profile_movies.append(movie_data)
    except Exception:
        flash("We couldn't analyze your favorites right now. Please try again.", "error")
        return redirect(url_for("home"))

    if not enriched_profile_movies:
        flash("We couldn't build a profile from your favorites right now. Please try again.", "error")
        return redirect(url_for("home"))

    user_profile_text, genre_counter, genre_names = build_user_preference_profile(enriched_profile_movies)

    if not user_profile_text:
        flash("Your current favorites do not yet contain enough descriptive metadata for recommendations.", "info")
        return redirect(url_for("home"))

    if not genre_counter:
        flash("We couldn't determine your top genre from your favorites yet.", "error")
        return redirect(url_for("home"))

    top_genre_id, _ = genre_counter.most_common(1)[0]
    top_genre_name = genre_names.get(
        top_genre_id,
        next((genre["name"] for genre in GENRES if genre["id"] == top_genre_id), "your favorite"),
    )
    profile_genre_names = [
        genre_names.get(
            genre_id,
            next((genre["name"] for genre in GENRES if genre["id"] == genre_id), None),
        )
        for genre_id, _ in genre_counter.most_common(3)
    ]
    profile_genre_names = [genre_name for genre_name in profile_genre_names if genre_name]

    candidate_movies = fetch_profile_candidate_movies(
        profile_movies=enriched_profile_movies,
        genre_counter=genre_counter,
        max_candidates=80,
    )
    if not candidate_movies:
        flash("We couldn't load recommendations right now. Please try again.", "error")
        return redirect(url_for("home"))

    favorited_movie_ids = {favorite.movie_id for favorite in favorites}
    recommended_movies = rank_movies_by_user_profile(
        user_profile_text=user_profile_text,
        candidate_movies=candidate_movies,
        excluded_movie_ids=favorited_movie_ids,
        top_k=10,
    )

    if not recommended_movies:
        flash("No highly similar recommendations were available from your current favorites right now. Please try again.", "info")
        return redirect(url_for("home"))

    return render_template(
        "recommendations.html",
        recommended_movies=recommended_movies,
        top_genre_name=top_genre_name,
        profile_genre_names=profile_genre_names,
        sampled_favorites_count=len(enriched_profile_movies),
        favorited_movie_ids=favorited_movie_ids,
    )

@app.route("/auth")
def auth():
    """Displays the authentication page for registration and sign-in actions.

    Session Authentication:
        Not required.

    Data Sources:
        Does not communicate with the TMDB API and does not query the local
        SQLite database.
    """
    return render_template("auth.html")


@app.route("/profile", methods=["GET", "POST"])
def profile():
    """Displays and updates the authenticated user's profile information.

    Session Authentication:
        Required.

    Data Sources:
        Queries and updates the local SQLite database and does not communicate
        with the TMDB API.
    """
    user_id = session.get("user_id")
    if not user_id:
        flash("Please sign in to view your profile.", "error")
        return redirect(url_for("auth"))

    user = User.query.get_or_404(user_id)

    if request.method == "POST":
        user.name = request.form.get("name", "").strip() or user.name
        user.favorite_genre = request.form.get("favorite_genre") or None
        # Commit the profile changes so the updated account metadata is stored
        # persistently in the local SQLite database.
        db.session.commit()
        flash("Profile updated successfully.", "success")
        return redirect(url_for("profile"))

    return render_template("profile.html", user=user, genres=GENRES)


@app.route("/add_favorite", methods=["POST"])
def add_favorite():
    """Stores a movie in the authenticated user's favorites collection.

    Session Authentication:
        Required.

    Data Sources:
        Queries and updates the local SQLite database and does not communicate
        with the TMDB API.
    """
    user_id = session.get("user_id")
    if not user_id:
        flash("Please sign in to save favorites.", "error")
        return redirect(url_for("auth"))

    movie_id = request.form.get("movie_id", type=int)
    title = request.form.get("title", "").strip()
    poster_path = request.form.get("poster_path", "").strip() or None

    existing_favorite = Favorite.query.filter_by(user_id=user_id, movie_id=movie_id).first()
    if existing_favorite:
        flash("That movie is already in your favorites.", "info")
        return redirect(url_for("movie_details", movie_id=movie_id))

    favorite = Favorite(
        user_id=user_id,
        movie_id=movie_id,
        title=title or "Untitled",
        poster_path=poster_path,
    )
    db.session.add(favorite)
    # Commit the newly created favorite record so it becomes immediately
    # available to the favorites and recommendation workflows.
    db.session.commit()

    flash("Movie added to favorites!", "success")
    return redirect(url_for("movie_details", movie_id=movie_id))


@app.route("/remove_favorite/<int:movie_id>", methods=["POST"])
def remove_favorite(movie_id):
    """Removes a movie from the authenticated user's favorites collection.

    Session Authentication:
        Required.

    Data Sources:
        Queries and updates the local SQLite database and does not communicate
        with the TMDB API.
    """
    user_id = session.get("user_id")
    if not user_id:
        flash("Please sign in to manage favorites.", "error")
        return redirect(url_for("auth"))

    favorite_record = Favorite.query.filter_by(user_id=user_id, movie_id=movie_id).first()
    if favorite_record:
        db.session.delete(favorite_record)
        # Commit the deletion so the removal is reflected in all future
        # favorites queries and recommendation calculations.
        db.session.commit()
        flash("Movie removed from favorites.", "info")

    return redirect(request.referrer or url_for("movie_details", movie_id=movie_id))


@app.route("/favorites")
def favorites():
    """Displays the authenticated user's locally stored favorite movies.

    Session Authentication:
        Required.

    Data Sources:
        Queries the local SQLite database and does not communicate with the
        TMDB API.
    """
    user_id = session.get("user_id")
    if not user_id:
        flash("Please sign in to view your favorites.", "error")
        return redirect(url_for("auth"))

    user_favorites = Favorite.query.filter_by(user_id=user_id).order_by(Favorite.id.desc()).all()
    return render_template("favorites.html", favorites=user_favorites)


@app.route("/submit_review/<int:movie_id>", methods=["POST"])
def submit_review(movie_id):
    """Stores a review submitted for a specific movie by an authenticated user.

    Session Authentication:
        Required.

    Data Sources:
        Updates the local SQLite database and does not communicate with the
        TMDB API.
    """
    user_id = session.get("user_id")
    if not user_id:
        flash("Please sign in to leave a review.", "error")
        return redirect(url_for("auth"))

    rating = request.form.get("rating", type=int)
    review_text = request.form.get("review_text", "").strip()
    movie_title = request.form.get("movie_title", "").strip() or "Untitled"

    if rating not in [1, 2, 3, 4, 5]:
        flash("Please choose a rating from 1 to 5 stars.", "error")
        return redirect(request.referrer or url_for("movie_details", movie_id=movie_id))

    if not review_text:
        flash("Please write a short review before submitting.", "error")
        return redirect(request.referrer or url_for("movie_details", movie_id=movie_id))

    review = Review(
        user_id=user_id,
        movie_id=movie_id,
        movie_title=movie_title,
        rating=rating,
        review_text=review_text,
    )
    db.session.add(review)
    # Commit the review insert so the new entry appears on the movie details
    # page and in the author's personal review history.
    db.session.commit()

    flash("Review submitted successfully.", "success")
    return redirect(request.referrer or url_for("movie_details", movie_id=movie_id))


@app.route("/my_reviews")
def my_reviews():
    """Displays the authenticated user's review history.

    Session Authentication:
        Required.

    Data Sources:
        Queries the local SQLite database and does not communicate with the
        TMDB API.
    """
    user_id = session.get("user_id")
    if not user_id:
        flash("Please sign in to view your reviews.", "error")
        return redirect(url_for("auth"))

    user_reviews = Review.query.filter_by(user_id=user_id).order_by(Review.created_at.desc()).all()
    return render_template("reviews.html", reviews=user_reviews)


@app.route("/register", methods=["POST"])
def register():
    """Registers a new user account and initializes an authenticated session.

    Session Authentication:
        Not required for access; successful completion creates session
        authentication for the new user.

    Data Sources:
        Queries and updates the local SQLite database and does not communicate
        with the TMDB API.
    """
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    favorite_genre = request.form.get("favorite_genre") or None

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        flash("An account with that email already exists.", "error")
        return redirect(url_for("auth"))

    password_hash = generate_password_hash(password)
    user = User(
        name=name,
        email=email,
        password_hash=password_hash,
        favorite_genre=favorite_genre,
    )
    db.session.add(user)
    # Commit the account creation so the user can be assigned a persistent
    # primary key before the session is initialized.
    db.session.commit()

    session["user_id"] = user.id
    flash("Account created successfully.", "success")
    return redirect(url_for("home"))


@app.route("/login", methods=["POST"])
def login():
    """Authenticates an existing user and stores the user identifier in session.

    Session Authentication:
        Not required for access; successful completion establishes session
        authentication.

    Data Sources:
        Queries the local SQLite database and does not communicate with the
        TMDB API.
    """
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password_hash, password):
        session["user_id"] = user.id
        flash("You are now signed in.", "success")
        return redirect(url_for("home"))

    flash("Invalid email or password.", "error")
    return redirect(url_for("auth"))


@app.route("/logout")
def logout():
    """Terminates the current authenticated session and redirects home.

    Session Authentication:
        Not strictly required, although it is intended for authenticated users.

    Data Sources:
        Does not communicate with the TMDB API and does not query the local
        SQLite database.
    """
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("home"))

with app.app_context():
    db.create_all()
    ensure_review_schema()

if __name__ == "__main__":
    app.run(debug=True)
