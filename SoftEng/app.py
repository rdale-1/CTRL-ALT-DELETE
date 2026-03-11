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
    if not API_KEY:
        return render_template("index.html", movies=[], genres=GENRES, current_page=1, total_pages=1, error="API_KEY not configured. Please check your .env file.")
    
    try:
        page = request.args.get("page", 1, type=int)
        url = f"{BASE_URL}/trending/movie/week?api_key={API_KEY}&page={page}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "status_code" in data and data["status_code"] != 200:
            return render_template("index.html", movies=[], genres=GENRES, current_page=1, total_pages=1, error=f"API Error: {data.get('status_message', 'Unknown error')}")
        
        return render_template("index.html", movies=data.get("results", []), genres=GENRES, current_page=page, total_pages=data.get("total_pages", 1))
    except Exception as e:
        return render_template("index.html", movies=[], genres=GENRES, current_page=1, total_pages=1, error=f"Error fetching movies: {str(e)}")

@app.route("/search")
def search():
    if not API_KEY:
        return render_template("index.html", movies=[], genres=GENRES, current_page=1, total_pages=1, error="API_KEY not configured. Please check your .env file.")
    
    try:
        query = request.args.get("q", "")
        page = request.args.get("page", 1, type=int)
        url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={query}&page={page}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "status_code" in data and data["status_code"] != 200:
            return render_template("index.html", movies=[], genres=GENRES, current_page=1, total_pages=1, error=f"API Error: {data.get('status_message', 'Unknown error')}")
        
        return render_template("index.html", movies=data.get("results", []), genres=GENRES, current_page=page, total_pages=data.get("total_pages", 1))
    except Exception as e:
        return render_template("index.html", movies=[], genres=GENRES, current_page=1, total_pages=1, error=f"Error searching movies: {str(e)}")

@app.route("/browse")
def browse():
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

    url = f"{BASE_URL}/discover/movie?api_key={API_KEY}"
    for k, v in params.items():
        url += f"&{k}={v}"

    try:
        response = requests.get(url, timeout=10)
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
    """Fetch and display detailed information for a specific movie."""
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
    
    return render_template("details.html", movie=movie_data)

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

@app.route("/auth")
def auth():
    return render_template("auth.html")

if __name__ == "__main__":
    app.run(debug=True)