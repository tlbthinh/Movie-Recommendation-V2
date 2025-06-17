# Movie Recommender System
This repository contains a movie recommendation system built using collaborative filtering techniques.
The system generates personalized movie recommendations based on user ratings, leveraging models such as KNN and Matrix Factorization.
The project utilizes the **[MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)**, which includes 1 million ratings from approximately 6,000 users on 4,000 movies.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)

## Prerequisites
- Python 3.8 or higher
- Jupyter Notebook (for running `analysis.ipynb`)
- Streamlit (for running the web application)
- Access to the MovieLens 1M Dataset (downloaded automatically or placed in the `data/` directory)

**Note**: As of 2025/04, the `scikit-surprise` library is incompatible with `numpy` versions >2.0. Ensure you use `numpy<2.0` to avoid compatibility issues.

## Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/tlbthinh/Movie-Recommendation-V2.git](https://github.com/tlbthinh/Movie-Recommendation-V2.git)
   cd Movie-Recommendation-V2

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application
1. Train Models: Open and run the `analysis.ipyn`b notebook to perform data analysis, train models (KNN, Matrix Factorization), and save trained models to the `checkpoint/` directory.
2. Launch the Web Application: Run the Streamlit app to interact with the recommender system
```bash
streamlit run main.py
```
<img src="https://github.com/tlbthinh/Movie-Recommendation-V2/blob/main/asset/demo_ui.gif" alt="Demo of Movie Recommender System" width="1000"/>
