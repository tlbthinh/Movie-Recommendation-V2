import streamlit as st
import plotly.express as px
import pandas as pd
from utils import load_data

def show_data_exploration():
    movies, ratings, users, exploded_genre, movie_ratings, _ = load_data()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Genre distribution",
        "🏆 Most rated genres",
        "🎭 Highest & lowest rated genres",
        "🎥 Top-K rated movies"
    ])

    with tab1:
        genre_counts = exploded_genre["genre"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Count"]

        fig = px.bar(
            genre_counts,
            x="Genre",
            y="Count",
            title="Number of movies per genre",
            labels={"Count": "Number of movies", "Genre": "Genre"},
            color="Count",
            color_continuous_scale="viridis"
        )

        fig.update_traces(texttemplate='', hoverinfo="x+y")
        fig.update_layout(xaxis_tickangle=-45, width=800, height=500)

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        genre_rating_count = movie_ratings.groupby("genre")["rating"].count().reset_index()
        genre_rating_count.columns = ["Genre", "Total Ratings"]

        top_genres = genre_rating_count.sort_values(by="Total Ratings", ascending=False).head(5)
        top_genres.reset_index(drop=True, inplace=True)

        st.dataframe(top_genres, hide_index=True, use_container_width=True)

    with tab3:
        genre_avg_ratings = movie_ratings.groupby("genre")["rating"].mean().reset_index()
        genre_avg_ratings = genre_avg_ratings.sort_values(by="rating", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top 3 highest-rated genres")
            st.dataframe(genre_avg_ratings.head(3), hide_index=True)

        with col2:
            st.subheader("Bottom 3 lowest-rated genres")
            st.dataframe(genre_avg_ratings.tail(3), hide_index=True)

    with tab4:
        k = st.slider("Select number of top movies to display", min_value=5, max_value=50, value=10)

        movie_ratings = movie_ratings.groupby(["movieId", "title"])["rating"].agg(["mean", "count"]).reset_index()
        movie_ratings = movie_ratings[movie_ratings["count"] >= 100]  # Filter movies with at least 100 ratings
        movie_ratings = movie_ratings.sort_values(by="mean", ascending=False)

        top_movies = movie_ratings[["title", "mean"]].head(k)
        top_movies.rename(columns={"title": "Tittle", "mean": "Rating"}, inplace=True)

        st.subheader(f"🎥 Top {k} movies with highest average rating (with at least 100 Ratings)")
        st.table(top_movies.set_index([pd.Index(range(1, len(top_movies) + 1))]))
