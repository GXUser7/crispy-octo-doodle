import os
import pickle
import sqlite3

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'recsys.db')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

database = sqlite3.connect(DB_PATH)
database.execute(
    'CREATE TABLE IF NOT EXISTS user_ratings '
    '(user_id INTEGER, movie_id INTEGER, rating REAL, PRIMARY KEY (user_id, movie_id))'
)
database.commit()
database.close()

app = FastAPI(title='Рекомендательная система фильмов')


def load_pickle(filename):
    path = os.path.join(MODELS_DIR, filename)
    with open(path, 'rb') as file:
        return pickle.load(file)


movies_table = load_pickle('movies.pkl').reset_index(drop=True)
ratings_table = load_pickle('users_pivot.pkl')
text_matrix = load_pickle('tfidf_matrix.pkl')
similar_movies_model = load_pickle('model_knn.pkl')
similar_users_model = load_pickle('model_user_knn.pkl')

movies_ratings_matrix = csr_matrix(ratings_table.values).T

movie_row_by_id = {}
for row_number in range(len(movies_table)):
    current_movie_id = int(movies_table['movieId'].iloc[row_number])
    movie_row_by_id[current_movie_id] = row_number

COMMANDS = {
    'GET /help': 'список всех команд API',
    'GET /movies?search=toy': 'поиск фильма по названию',
    'GET /movies/{movie_id}': 'подробная информация о фильме',
    'GET /genres': 'список всех жанров',
    'GET /popular?n=10&genre=Comedy&min_year=2000': 'топ популярных фильмов (с фильтром по жанру и году)',
    'GET /recommend/content/{movie_id}?n=10': 'похожие фильмы по описанию (контентная фильтрация)',
    'GET /recommend/similar/{movie_id}?n=10': 'похожие фильмы по оценкам зрителей (KNN)',
    'GET /recommend/user/{user_id}?n=10': 'рекомендации по 20 похожим пользователям (KNN)',
    'GET /recommend/hybrid/{user_id}?n=10': 'гибридные рекомендации (коллаборативная + контентная)',
    'GET /recommend/by-rating/{movie_id}?rating=5&user_id=0': '5 рекомендаций сразу после оценки фильма',
    'POST /rate': 'сохранить оценку в базу: {"user_id": 672, "movie_id": 1, "rating": 5.0}',
    'GET /users/new': 'получить новый свободный ID пользователя',
    'GET /users/{user_id}/ratings': 'все оценки выбранного пользователя',
}


def find_movie_row(movie_id):
    if movie_id not in movie_row_by_id:
        raise HTTPException(status_code=404, detail='Фильм не найден')
    return movie_row_by_id[movie_id]


def get_best_rows(scores, count):
    sorted_rows = np.argsort(scores)[::-1]
    return sorted_rows[:count]


def get_user_ratings(user_id):
    ratings = {}

    if user_id in ratings_table.index:
        user_row = ratings_table.loc[user_id]
        for current_movie_id, rating_value in user_row.items():
            if rating_value > 0:
                ratings[int(current_movie_id)] = float(rating_value)

    database = sqlite3.connect(DB_PATH)
    saved_rows = database.execute(
        'SELECT movie_id, rating FROM user_ratings WHERE user_id = ?',
        (user_id,)
    ).fetchall()
    database.close()

    for current_movie_id, rating_value in saved_rows:
        ratings[int(current_movie_id)] = float(rating_value)

    return ratings


def make_user_ratings_array(user_id):
    user_ratings_array = np.zeros(len(movies_table), dtype='float32')
    ratings = get_user_ratings(user_id)
    for current_movie_id, rating_value in ratings.items():
        if current_movie_id in movie_row_by_id:
            row_number = movie_row_by_id[current_movie_id]
            user_ratings_array[row_number] = rating_value
    return user_ratings_array


def make_movie_list(row_numbers, scores):
    result = []
    for position in range(len(row_numbers)):
        row_number = int(row_numbers[position])
        score = float(scores[position])
        movie = movies_table.iloc[row_number]
        result.append({
            'movie_id': int(movie['movieId']),
            'title': movie['title'],
            'year': movie['year'],
            'genres': movie['genres'],
            'score': round(score, 3),
        })
    return result


def make_recommendation_card(movie, score, reason):
    percent = int(round(float(score) * 100))
    if percent < 1:
        percent = 1
    if percent > 99:
        percent = 99

    return {
        'movie_id': int(movie['movieId']),
        'title': movie['title'],
        'year': movie['year'],
        'genres': movie['genres'],
        'score': round(float(score), 3),
        'confidence_percent': percent,
        'reason': reason,
    }


def has_same_genre(first_genres_text, second_genres_text):
    if not first_genres_text or not second_genres_text:
        return False

    first_genres = first_genres_text.split(', ')
    second_genres = second_genres_text.split(', ')
    for genre in first_genres:
        if genre in second_genres:
            return True
    return False


def get_neighbors_average_scores(user_ratings_array, exclude_user_id, num_neighbors=20):
    ask_count = min(num_neighbors + 1, len(ratings_table))
    distances, neighbor_rows = similar_users_model.kneighbors(
        csr_matrix([user_ratings_array]),
        n_neighbors=ask_count,
    )

    neighbors = []
    for position in range(len(neighbor_rows[0])):
        neighbor_row = int(neighbor_rows[0][position])
        if ratings_table.index[neighbor_row] == exclude_user_id:
            continue
        neighbors.append(neighbor_row)
        if len(neighbors) == num_neighbors:
            break

    mean_scores = ratings_table.iloc[neighbors].mean().values
    return mean_scores.copy()


def get_item_rating(item):
    return item['rating']


@app.get('/help')
def show_help():
    return COMMANDS


@app.get('/movies')
def search_movies(search: str = '', limit: int = 30):
    matches = movies_table['title'].str.contains(search, case=False, regex=False)
    found_movies = movies_table[matches].sort_values('count_rating', ascending=False).head(limit)

    result = []
    for _, movie in found_movies.iterrows():
        result.append({
            'movie_id': int(movie['movieId']),
            'title': movie['title'],
            'year': movie['year'],
        })
    return result


@app.get('/movies/{movie_id}')
def get_movie(movie_id: int):
    movie_row = find_movie_row(movie_id)
    movie = movies_table.iloc[movie_row]
    return {
        'movie_id': int(movie['movieId']),
        'title': movie['title'],
        'year': movie['year'],
        'genres': movie['genres'],
        'overview': movie['overview'],
        'avg_rating': round(float(movie['avg_rating']), 2),
        'count_rating': int(movie['count_rating']),
        'w_score': round(float(movie['w_score']), 3),
    }


@app.get('/genres')
def get_genres():
    genres = set()
    for genres_text in movies_table['genres']:
        if genres_text:
            for genre_name in genres_text.split(', '):
                genres.add(genre_name)
    return sorted(genres)


@app.get('/popular')
def popular(n: int = 10, genre: str = '', min_year: int = 0):
    filtered_movies = movies_table.copy()

    if genre:
        matches = filtered_movies['genres'].str.contains(genre, case=False, regex=False)
        filtered_movies = filtered_movies[matches]

    if min_year > 0:
        years = pd.to_numeric(filtered_movies['year'], errors='coerce').fillna(0)
        filtered_movies = filtered_movies[years >= min_year]

    best_movies = filtered_movies.sort_values('w_score', ascending=False).head(n)
    return make_movie_list(best_movies.index.tolist(), best_movies['w_score'].tolist())


@app.get('/recommend/content/{movie_id}')
def recommend_content(movie_id: int, n: int = 10):
    movie_row = find_movie_row(movie_id)
    similarities = cosine_similarity(text_matrix[movie_row], text_matrix)[0]
    similarities[movie_row] = -1
    best_rows = get_best_rows(similarities, n)
    return make_movie_list(best_rows, similarities[best_rows])


@app.get('/recommend/similar/{movie_id}')
def recommend_similar(movie_id: int, n: int = 10):
    movie_row = find_movie_row(movie_id)
    ask_count = min(n + 1, len(movies_table))
    distances, neighbor_rows = similar_movies_model.kneighbors(
        movies_ratings_matrix[movie_row],
        n_neighbors=ask_count,
    )

    found_rows = []
    found_scores = []
    for position in range(len(neighbor_rows[0])):
        candidate_row = int(neighbor_rows[0][position])
        if candidate_row == movie_row:
            continue
        found_rows.append(candidate_row)
        found_scores.append(1.0 - float(distances[0][position]))
        if len(found_rows) == n:
            break

    return make_movie_list(found_rows, found_scores)


@app.get('/recommend/user/{user_id}')
def recommend_user(user_id: int, n: int = 10):
    user_ratings_array = make_user_ratings_array(user_id)
    if user_ratings_array.sum() == 0:
        return popular(n=n)

    scores = get_neighbors_average_scores(user_ratings_array, exclude_user_id=user_id, num_neighbors=20)
    scores[user_ratings_array > 0] = -1
    best_rows = get_best_rows(scores, n)
    return make_movie_list(best_rows, scores[best_rows])


@app.get('/recommend/hybrid/{user_id}')
def recommend_hybrid(user_id: int, n: int = 10):
    user_ratings_array = make_user_ratings_array(user_id)
    if user_ratings_array.sum() == 0:
        return popular(n=n)

    neighbors_scores = get_neighbors_average_scores(user_ratings_array, exclude_user_id=user_id, num_neighbors=20)
    if neighbors_scores.max() > 0:
        neighbors_scores = neighbors_scores / neighbors_scores.max()

    liked_rows = np.where(user_ratings_array >= 4)[0]
    if len(liked_rows) == 0:
        liked_rows = [int(np.argmax(user_ratings_array))]

    text_scores = np.zeros(len(movies_table))
    for liked_row in liked_rows:
        text_scores = text_scores + cosine_similarity(text_matrix[int(liked_row)], text_matrix)[0]
    text_scores = text_scores / len(liked_rows)

    hybrid_scores = 0.5 * neighbors_scores + 0.5 * text_scores
    hybrid_scores[user_ratings_array > 0] = -1
    best_rows = get_best_rows(hybrid_scores, n)
    return make_movie_list(best_rows, hybrid_scores[best_rows])


@app.get('/recommend/by-rating/{movie_id}')
def recommend_by_rating(movie_id: int, rating: float, user_id: int = 0, n: int = 5):
    movie_row = find_movie_row(movie_id)
    user_ratings_array = make_user_ratings_array(user_id)

    if rating >= 4:
        ratings_similarity = cosine_similarity(movies_ratings_matrix[movie_row], movies_ratings_matrix)[0]
        text_similarity = cosine_similarity(text_matrix[movie_row], text_matrix)[0]
        total_scores = 0.5 * ratings_similarity + 0.5 * text_similarity
        total_scores[movie_row] = -1
        total_scores[user_ratings_array > 0] = -1

        result = []
        for best_row in get_best_rows(total_scores, n):
            movie = movies_table.iloc[int(best_row)]
            result.append(
                make_recommendation_card(movie, total_scores[best_row], 'Похож по оценкам зрителей и описанию')
            )
        return result

    disliked_genres = movies_table.iloc[movie_row]['genres']
    popular_movies = movies_table.sort_values('w_score', ascending=False)

    result = []
    for candidate_row, movie in popular_movies.iterrows():
        if candidate_row == movie_row or user_ratings_array[candidate_row] > 0:
            continue
        if has_same_genre(disliked_genres, movie['genres']):
            continue

        result.append(
            make_recommendation_card(movie, float(movie['w_score']) / 5.0, 'Популярный фильм из других жанров')
        )
        if len(result) == n:
            break

    return result


class Rating(BaseModel):
    user_id: int
    movie_id: int
    rating: float


@app.post('/rate')
def rate(item: Rating):
    find_movie_row(item.movie_id)
    if item.rating < 0.5 or item.rating > 5:
        raise HTTPException(status_code=400, detail='Оценка должна быть от 0.5 до 5')

    database = sqlite3.connect(DB_PATH)
    database.execute(
        'INSERT OR REPLACE INTO user_ratings VALUES (?, ?, ?)',
        (item.user_id, item.movie_id, item.rating)
    )
    database.commit()
    database.close()
    return {'status': 'ok'}


@app.get('/users/new')
def new_user():
    database = sqlite3.connect(DB_PATH)
    max_id_in_database = database.execute('SELECT MAX(user_id) FROM user_ratings').fetchone()[0] or 0
    database.close()
    max_id_in_table = int(ratings_table.index.max())
    return {'user_id': int(max(max_id_in_table, max_id_in_database)) + 1}


@app.get('/users/{user_id}/ratings')
def user_ratings(user_id: int):
    ratings = get_user_ratings(user_id)
    result = []
    for current_movie_id, rating_value in ratings.items():
        if current_movie_id in movie_row_by_id:
            movie = movies_table.iloc[movie_row_by_id[current_movie_id]]
            result.append({
                'movie_id': int(movie['movieId']),
                'title': movie['title'],
                'year': movie['year'],
                'genres': movie['genres'],
                'rating': float(rating_value),
            })

    result.sort(key=get_item_rating, reverse=True)
    return result


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
