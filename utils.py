import pickle
import re

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Movie Dashboard",
    page_icon="🎥",
    layout="wide"
)

with open('checkpoint/svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)
with open('checkpoint/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

@st.cache_data
def load_data():
    movies = pd.read_csv('./data/movies.dat', sep='::', engine='python', \
                         encoding='latin-1', header=None, \
                         names=["movieId", "title", "genre"])
    ratings = pd.read_csv('./data/ratings.dat', sep='::', engine='python', encoding='latin-1',
                          header=None, names=['userId', 'movieId', 'rating', 'timestamp'])
    users = pd.read_csv('./data/users.dat', sep='::', engine='python', \
                        encoding='latin-1', header=None, \
                        names=["userId", "gender", "age", "occupation", "zipcode"])
    image_data = pd.read_csv('data/ml1m_images.csv')

    movies["genre"] = movies["genre"].str.split("|")
    exploded_genre = movies.explode("genre")
    movie_ratings = pd.merge(ratings, exploded_genre, on="movieId")
    return movies, ratings, users, exploded_genre, movie_ratings, image_data

def extract_year(title):
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return match.group(1)
    else:
        return None

def render_star_rating(rating):
    full_stars = int(rating)
    has_half_star = 0
    if (rating - full_stars) >= 0.25:
        if rating - full_stars >= 0.75:
            full_stars += 1
        else:
            has_half_star = 1

    empty_stars = 5 - full_stars - has_half_star

    stars = "★" * full_stars
    if has_half_star:
        stars += "⯨"
    stars += "☆" * empty_stars

    return f'<span style="font-size:24px;">Ratings: </span> ' \
           f'<span style="font-size:24px; color:#FFD700;">{stars}</span>'