import os # Add this
from flask import Flask, render_template, request
import requests
from dotenv import load_dotenv # Add this (make sure to run 'pip install python-dotenv' in your terminal)

# Load the .env file
load_dotenv()

app = Flask(__name__)

# Change line 6 to this:
API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

@app.route("/")
def home():
    url = f"{BASE_URL}/trending/movie/week?api_key={API_KEY}"
    data = requests.get(url).json()
    return render_template("index.html", movies=data["results"])

@app.route("/search")
def search():
    query = request.args.get("q")
    url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={query}"
    data = requests.get(url).json()
    return render_template("index.html", movies=data["results"])

if __name__ == "__main__":
    app.run(debug=True)
