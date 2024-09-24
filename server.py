"""

Review Analyzer Server

This server processes restaurant reviews, analyzes their sentiment, and provides an API to query and submit reviews.
The main functionalities of the server include:
1. Loading reviews from a CSV file and alaysing their sentiment using the VADER sentiment analysis tool.
2. Providing a GET endpoint to filter and retrieve all reviews based on location, and date range.
3. Providing a POST endpoint to submit a new review.

Approach:
1. Load the reviews from the CSV file.
2. Implement a GET method to filter reviews based on location and date range.
3. Implement a POST method to accept new reviews, generate necessary metadata (Review ID, Timestamp), and analyze their sentiment, and store them.

Additional Features:
- Extracting and counting the most common adjective-noun pairs from the reviews.
- Filtering out stop words from the reviews.

"""
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        self.reviews = reviews
        self.allowed_locations = {
            "Albuquerque, New Mexico",
            "Carlsbad, California",
            "Chula Vista, California",
            "Colorado Springs, Colorado",
            "Denver, Colorado",
            "El Cajon, California",
            "El Paso, Texas",
            "Escondido, California",
            "Fresno, California",
            "La Mesa, California",
            "Las Vegas, Nevada",
            "Los Angeles, California",
            "Oceanside, California",
            "Phoenix, Arizona",
            "Sacramento, California",
            "Salt Lake City, Utah",
            "San Diego, California",
            "Tucson, Arizona",
        }

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """
        method = environ["REQUEST_METHOD"]
        if method == "GET":
            return self.handle_get(environ, start_response)
        elif method == "POST":
            return self.handle_post(environ, start_response)
        else:
            start_response("405 Method Not Allowed", [("Content-Type", "text/plain")])
            return [b"Method Not Allowed"]

    def handle_get(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        query_string = environ.get("QUERY_STRING", "")
        params = parse_qs(query_string)

        location = params.get("location", [None])[0]
        start_date = params.get("start_date", [None])[0]
        end_date = params.get("end_date", [None])[0]

        filtered_reviews = self.reviews
        if location:
            if location not in self.allowed_locations:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [json.dumps({"error": "Invalid location"}).encode("utf-8")]
            filtered_reviews = [review for review in filtered_reviews if review["Location"] == location]

        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review["Timestamp"], '%Y-%m-%d %H:%M:%S') >= start_date]

        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review["Timestamp"], '%Y-%m-%d %H:%M:%S') <= end_date]

        for review in filtered_reviews:
            review["sentiment"] = self.analyze_sentiment(review["ReviewBody"])

        filtered_reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

        response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
        start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
        ])
        return [response_body]

    def handle_post(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        try:
            request_body_size = int(environ.get('CONTENT_LENGTH', 0))
        except (ValueError):
            request_body_size = 0

        request_body = environ['wsgi.input'].read(request_body_size).decode('utf-8')
        params = parse_qs(request_body)

        location = params.get("Location", [None])[0]
        review_body = params.get("ReviewBody", [None])[0]

        if not location or not review_body:
            start_response("400 Bad Request", [("Content-Type", "application/json")])
            return [json.dumps({"error": "Location and ReviewBody are required"}).encode("utf-8")]

        if location not in self.allowed_locations:
            start_response("400 Bad Request", [("Content-Type", "application/json")])
            return [json.dumps({"error": "Invalid location"}).encode("utf-8")]

        new_review = {
            "ReviewId": str(uuid.uuid4()),
            "Location": location,
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "ReviewBody": review_body,
            "sentiment": self.analyze_sentiment(review_body)
        }

        self.reviews.append(new_review)

        response_body = json.dumps(new_review).encode("utf-8")
        start_response("201 Created", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
        ])
        return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
