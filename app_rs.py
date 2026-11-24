import pandas as pd
import requests
import streamlit as st

API = 'http://127.0.0.1:8000'

st.set_page_config(page_title='Рекомендательная система фильмов', layout='wide')
st.title('Рекомендательная система фильмов')

try:
    requests.get(API + '/help', timeout=3)
except requests.ConnectionError:
    st.error('API не запущен. Сначала запустите api.py, а потом обновите страницу.')
    st.stop()

if 'user_id' not in st.session_state:
    st.session_state.user_id = requests.get(API + '/users/new').json()['user_id']

user_id = int(
    st.sidebar.number_input(
        'Ваш ID пользователя',
        min_value=1,
        value=st.session_state.user_id,
        step=1,
    )
)
st.sidebar.caption('ID 1-671 - пользователи из датасета. ID от 672 - новые пользователи сайта.')

menu = st.sidebar.radio(
    'Выберите режим',
    [
        'Оценить фильм',
        'Топ популярных и фильтр на знаниях',
        'По описанию (контентная фильтрация)',
        'Похожие фильмы (коллаборативная KNN)',
        'По похожим людям (коллаборативная KNN)',
        'Персональные (гибридный подход)',
        'Мои оценки',
        'Справка',
    ],
)


def choose_movie(field_key):
    search_text = st.text_input('Поиск фильма (название на английском)', key=field_key)
    found_movies = requests.get(API + '/movies', params={'search': search_text}).json()
    if not found_movies:
        st.warning('Ничего не найдено')
        return None

    movie_choices = {}
    for movie in found_movies:
        label = f"{movie['title']} ({movie['year']}) [id: {movie['movie_id']}]"
        movie_choices[label] = movie['movie_id']

    selected_label = st.selectbox(
        'Выберите фильм из найденных',
        list(movie_choices.keys()),
        key=field_key + '_list',
    )
    return movie_choices[selected_label]


def show_table(movies, score_name='Балл'):
    if not movies:
        st.info('Список пуст.')
        return

    table = pd.DataFrame(movies)
    table = table.rename(columns={
        'title': 'Фильм',
        'year': 'Год',
        'genres': 'Жанры',
        'score': score_name,
        'rating': score_name,
    })
    st.dataframe(table[['Фильм', 'Год', 'Жанры', score_name]], hide_index=True, use_container_width=True)


if menu == 'Оценить фильм':
    st.write('Выберите фильм и поставьте оценку. Система сохранит ее в базу данных и сразу покажет 5 рекомендаций.')
    movie_id = choose_movie('rate')
    if movie_id:
        movie = requests.get(f'{API}/movies/{movie_id}').json()
        st.subheader(f"{movie['title']} ({movie['year']})")
        st.caption(
            f"Жанры: {movie['genres']} | Средняя оценка: {movie['avg_rating']} | Всего оценок: {movie['count_rating']}"
        )
        st.write(movie['overview'])

        rating = st.slider('Ваша оценка', 0.5, 5.0, 4.0, 0.5)
        if st.button('Сохранить оценку', type='primary'):
            requests.post(
                API + '/rate',
                json={'user_id': user_id, 'movie_id': movie_id, 'rating': rating},
            )
            st.success(f'Оценка {rating} сохранена в базу для пользователя {user_id}!')

            recommendations = requests.get(
                f'{API}/recommend/by-rating/{movie_id}',
                params={'rating': rating, 'user_id': user_id},
            ).json()

            if rating >= 4:
                st.subheader('Фильм понравился - вот 5 похожих рекомендаций:')
            else:
                st.subheader('Фильм не понравился - вот что может подойти вместо него:')

            if not recommendations:
                st.info('Рекомендации не найдены.')
            else:
                columns = st.columns(len(recommendations))
                for position in range(len(recommendations)):
                    card = recommendations[position]
                    with columns[position]:
                        st.markdown(f"**{card['title']}** ({card['year']})")
                        st.caption(card['genres'])
                        percent = card['confidence_percent']
                        st.metric('Уверенность', f'{percent}%')
                        st.progress(percent / 100.0)
                        st.caption(card['reason'])

elif menu == 'Топ популярных и фильтр на знаниях':
    st.caption('Взвешенный рейтинг IMDb (w_score) плюс фильтрация на знаниях по жанру и году выхода.')
    genres = requests.get(API + '/genres').json()
    left_column, right_column = st.columns(2)
    with left_column:
        genre = st.selectbox('Жанр', ['Все'] + genres)
    with right_column:
        min_year = st.slider('Фильмы не старше года (0 - любой год)', 0, 2016, 0, 1)

    query_params = {'min_year': min_year}
    if genre != 'Все':
        query_params['genre'] = genre

    movies_data = requests.get(API + '/popular', params=query_params).json()
    show_table(movies_data, 'Взвешенный рейтинг')

elif menu == 'По описанию (контентная фильтрация)':
    st.caption('Подбирает фильмы с похожими жанрами и сюжетом через TF-IDF и косинусное сходство.')
    movie_id = choose_movie('content')
    if movie_id:
        movies_data = requests.get(f'{API}/recommend/content/{movie_id}').json()
        show_table(movies_data, 'Похожесть')

elif menu == 'Похожие фильмы (коллаборативная KNN)':
    st.caption('Ищет фильмы, которые зрители оценивали похожим образом (модель NearestNeighbors).')
    movie_id = choose_movie('similar')
    if movie_id:
        movies_data = requests.get(f'{API}/recommend/similar/{movie_id}').json()
        show_table(movies_data, 'Похожесть')

elif menu == 'По похожим людям (коллаборативная KNN)':
    st.caption('Ищет 20 пользователей с похожими оценками (NearestNeighbors). Если оценок нет - срабатывает холодный старт.')
    movies_data = requests.get(f'{API}/recommend/user/{user_id}').json()
    show_table(movies_data, 'Балл')

elif menu == 'Персональные (гибридный подход)':
    st.caption('Объединяет коллаборативную фильтрацию (50%) и контентную фильтрацию (50%). Если оценок нет - показывает популярное.')
    movies_data = requests.get(f'{API}/recommend/hybrid/{user_id}').json()
    show_table(movies_data, 'Гибридный балл')

elif menu == 'Мои оценки':
    user_ratings_list = requests.get(f'{API}/users/{user_id}/ratings').json()
    if user_ratings_list:
        show_table(user_ratings_list, 'Оценка')
    else:
        st.info('У этого пользователя пока нет оценок.')

elif menu == 'Справка':
    st.markdown('''
**Основные режимы сайта**
1. **Оценить фильм** - выберите фильм, поставьте оценку и сразу получите 5 рекомендаций с уверенностью.
2. **Топ популярных и фильтр на знаниях** - лучшие фильмы по формуле IMDb, можно задать жанр и год выпуска.
3. **По описанию (контентная фильтрация)** - подбор фильмов со схожим описанием и жанрами (TF-IDF).
4. **Похожие фильмы (коллаборативная KNN)** - похожие фильмы по оценкам зрителей (NearestNeighbors).
5. **По похожим людям (коллаборативная KNN)** - рекомендации от 20 пользователей с похожим вкусом.
6. **Персональные (гибридный подход)** - комбинация коллаборативного и контентного подходов.
7. **Мои оценки** - список всех фильмов, которые оценил выбранный ID.
''')
    st.markdown(f'**Команды API** ({API}, документация: {API}/docs)')
    commands = requests.get(API + '/help').json()
    for command, description in commands.items():
        st.markdown(f'- `{command}` - {description}')
