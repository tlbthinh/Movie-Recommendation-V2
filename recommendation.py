import random
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_data, knn_model, svd_model, extract_year, render_star_rating


def knn_recommend_movies(movies, ratings, movie_name, k):
    movie_match = movies[movies['title'] == movie_name]
    if movie_match.empty:
        return []
    movie_id = movie_match['movieId'].values[0]
    try:
        iid = knn_model.trainset.to_inner_iid(movie_id)
    except ValueError:
        recommendations = recommend_popular_movies(ratings, movies)
        return random.sample(recommendations, min(k, len(recommendations)))

    neighbors = knn_model.get_neighbors(iid, k)
    similar_movie_ids = [knn_model.trainset.to_raw_iid(inner_id) for inner_id in neighbors]

    recommendations = []
    for similar_movie_id in similar_movie_ids:
        temp_df = movies[movies['movieId'] == similar_movie_id]
        if not temp_df.empty:
            recommendations.append({
                "movieId": similar_movie_id,
                "title": temp_df['title'].values[0],
                "genre": temp_df['genre'].values[0] if 'genre' in temp_df.columns else "",
                "avg_rating": round(ratings[ratings['movieId'] == similar_movie_id]['rating'].mean(), 1),
                "num_ratings": ratings[ratings['movieId'] == similar_movie_id].shape[0]
            })

    return recommendations

def svd_recommend_movies(movies, ratings, movie_name, k):
    movie_match = movies[movies['title'] == movie_name]
    if movie_match.empty:
        return []
    movie_id = movie_match['movieId'].values[0]
    try:
        iid = svd_model.trainset.to_inner_iid(movie_id)
    except:
        recommendations = recommend_popular_movies(ratings, movies)
        return random.sample(recommendations, min(k, len(recommendations)))

    qi = svd_model.qi
    target_factors = qi[iid].reshape(1, -1)
    similarities = cosine_similarity(target_factors, qi)[0]
    similar_indices = similarities.argsort()[::-1]
    similar_iids = [idx for idx in similar_indices if idx != iid][:k]
    similar_movie_ids = [svd_model.trainset.to_raw_iid(iid) for iid in similar_iids]
    recommendations = []
    for movie_id in similar_movie_ids:
        temp_df = movies[movies['movieId'] == movie_id]
        if not temp_df.empty:
            recommendations.append({
                "movieId": movie_id,
                "title": temp_df['title'].values[0],
                "genre": temp_df['genre'].values[0] if 'genre' in temp_df.columns else "",
                "avg_rating": round(ratings[ratings['movieId'] == movie_id]['rating'].mean(), 1),
                "num_ratings": ratings[ratings['movieId'] == movie_id].shape[0]
            })
    return recommendations

# As we filtered out some movies, we need a fallback function to handle when a movie is missing from training data
# --> We recommend popular movies instead - top 100 highest-rated movies (>= 20 rating counts)
# Then in the recommend_movies function, we randomly pick `k` movies.
@st.cache_data
def recommend_popular_movies(ratings, movies, k=100):
    movie_stats = ratings.groupby("movieId").agg(avg_rating=("rating", "mean"), num_ratings=("rating", "count"))
    top_movies = movie_stats[movie_stats["num_ratings"] >= 20].nlargest(k, "avg_rating").index
    popular_movies_df = movies[movies['movieId'].isin(top_movies)]
    recommendations = []
    for _, row in popular_movies_df.iterrows():
        recommendations.append({
            "movieId": row["movieId"],
            "title": row["title"],
            "genre": row.get("genre", ""),  # Fallback to empty string if no genre
            "avg_rating": round(movie_stats.loc[row["movieId"], "avg_rating"], 1),
            "num_ratings": movie_stats.loc[row["movieId"], "num_ratings"]
        })

    return recommendations

def recommend_movies(model_name, movies, ratings, movie_name, k):
    if model_name == "KNN":
        return knn_recommend_movies(movies, ratings, movie_name, k)
    elif model_name == "Matrix factorization":
        return svd_recommend_movies(movies, ratings, movie_name, k)
    return []


def show_popular_movies_grid(movies, ratings, image_data):
    st.subheader("🎬 Popular Movies")
    popular_movies = recommend_popular_movies(ratings, movies)
    cols_per_row = 5
    rows = [popular_movies[i:i + cols_per_row] for i in range(0, len(popular_movies), cols_per_row)]
    for row in rows:
        cols = st.columns(cols_per_row)
        for col, movie in zip(cols, row):
            with col:
                movie_image = image_data[image_data['item_id'] == movie['movieId']]['image'].values
                movie_image_url = movie_image[0] if len(
                    movie_image) > 0 else "https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg"
                st.image(movie_image_url, use_container_width=True)
                st.caption(movie['title'])

def show_recommendation():
    st.title("Movie Recommender System")
    movies, ratings, users, exploded_genre, movie_ratings, image_data = load_data()
    selected_movie = st.selectbox("Select a Movie", movies['title'].unique())
    model_name = st.selectbox("Choose recommendation model", ["KNN", "Matrix factorization"])
    k = st.slider("Number of Recommendations", min_value=1, max_value=20, value=10, step=1)
    if st.button("Recommend"):
        st.session_state['recommend_clicked'] = True
        recommendations = recommend_movies(model_name, movies, ratings, selected_movie.strip(), k)
        if recommendations:
            for i, movie in enumerate(recommendations):
                movie_image = image_data[image_data['item_id'] == movie['movieId']]['image'].values
                if len(movie_image) > 0:
                    movie_image_url = movie_image[0]
                else:
                    movie_image_url = "https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg"
                col1, col2 = st.columns([0.5, 0.5])
                with col1:
                    st.header(movie['title'])
                    year_text = "Year of Publication: Unknown"
                    year = extract_year(movie['title'])
                    if year:
                        year_text = f"Year of Publication: {year}"
                    st.markdown(f'<span style="font-size:24px;">{year_text}</span>', unsafe_allow_html=True)
                    st.markdown(render_star_rating(movie['avg_rating']), unsafe_allow_html=True)
                    st.markdown(f'<span style="font-size:24px;">{movie['num_ratings']}+ Reviews</span>', unsafe_allow_html=True)
                with col2:
                    st.image(movie_image_url)

                st.markdown("""---""")

    if not st.session_state.get('recommend_clicked', False):
        show_popular_movies_grid(movies, ratings, image_data)
