import os
from datetime import datetime, timedelta
from flask import Flask, render_template, request
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

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

@app.route("/")
def home():
    page = request.args.get("page", 1, type=int)
    url = f"{BASE_URL}/trending/movie/week?api_key={API_KEY}&page={page}"
    data = requests.get(url).json()
    return render_template("index.html", movies=data.get("results", []), genres=GENRES, current_page=page, total_pages=data.get("total_pages", 1))

@app.route("/search")
def search():
    query = request.args.get("q", "")
    page = request.args.get("page", 1, type=int)
    url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={query}&page={page}"
    data = requests.get(url).json()
    return render_template("index.html", movies=data.get("results", []), genres=GENRES, current_page=page, total_pages=data.get("total_pages", 1))

@app.route("/browse")
def browse():
    page = request.args.get("page", 1, type=int)
    year     = request.args.get("year")
    rating   = request.args.get("rating")
    popular  = request.args.get("popular")
    genre_id = request.args.get("genre")

    params = {"page": page}

    if year:
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

    url = f"{BASE_URL}/discover/movie?api_key={API_KEY}"
    for k, v in params.items():
        url += f"&{k}={v}"

    data = requests.get(url).json()

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
    )

@app.route("/api/search-suggestions")
def search_suggestions():
    query = request.args.get("q", "")
    
    if len(query) < 2:
        return {"suggestions": []}
    
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
    import random
    page = random.randint(1, 50)
    url = f"{BASE_URL}/trending/movie/week?api_key={API_KEY}&page={page}"
    data = requests.get(url).json()
    movies = data.get("results", [])
    
    if not movies:
        return {"movie": None}
    
    random_movie = random.choice(movies)
    return {"movie": random_movie}

if __name__ == "__main__":
    app.run(debug=True)