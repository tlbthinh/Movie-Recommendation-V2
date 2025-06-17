# Movie Recommender System
This repository contains a movie recommendation system built using collaborative filtering techniques.
The system provides movie recommendations based on user ratings, with various models like KNN and Matrix Factorization.
This project uses the **[MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)**

## Project Structure
- `checkpoint/`: Contains trained models for easy reloading and inference.
- `data/`: Contains CSV files to load movies, user, ratings and image urls.
- `analysis.ipynb`: Jupyter notebook used to train models, perform data analysis, and evaluate recommendation strategies.
- `main.py`: Python script that runs the Streamlit web application for the movie recommender system.

## Install Dependencies
2025/04: `scikit-surprise` is not compatible with `numpy` versions greater than 2.0. To resolve this, you should install `numpy` with a version lower than 2.0.

```bash
pip install -r requirements.txt

![image](https://github.com/user-attachments/assets/d00cf296-71df-4f5f-a366-ffb3cae312f1)

