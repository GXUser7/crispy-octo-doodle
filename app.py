# -*- coding: utf-8 -*-
"""
Приложение для прогноза стоимости недвижимости.

Запуск:  streamlit run app.py
"""
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

APP_VERSION = '1.0'

# пути считаем от самого файла, чтобы приложение запускалось из любой папки
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# В облаке большие файлы лежат в постоянном хранилище (/data), а не в репозитории.
# Локально этой папки нет, поэтому берём файлы рядом с app.py.
PERSIST_DIR = os.environ.get('APP_DATA_DIR', '/data')


def find_file(name):
    in_persist = os.path.join(PERSIST_DIR, name)
    if os.path.exists(in_persist):
        return in_persist
    return os.path.join(BASE_DIR, name)


MODEL_FILE = find_file('model.pkl')
DATA_FILE = find_file('app_data.csv')
METRICS_FILE = find_file('app_metrics.csv')

st.set_page_config(page_title='Оценка стоимости недвижимости',
                   page_icon='🏠', layout='wide')


# ----------------------------------------------------------------------
# Загрузка модели и данных.
# cache_resource и cache_data выполняются ОДИН раз при старте приложения,
# а не на каждый запрос пользователя.
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    return pd.read_csv(DATA_FILE)


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_FILE):
        return None
    return pd.read_csv(METRICS_FILE, index_col=0)


package = load_model()
data = load_data()
metrics = load_metrics()

if package is None:
    st.error('Не найден файл model.pkl. Сначала запустите notebook '
             'real_estate_price.ipynb целиком — он обучит модель и сохранит её.')
    st.stop()


# ----------------------------------------------------------------------
# Сборка признаков — ровно так же, как при обучении
# ----------------------------------------------------------------------
def make_features(size, rooms, halls, building_age, total_floors, floor,
                  sub_type, heating_type, city, county):
    total_rooms = rooms + halls

    row = {
        'size': size,
        'building_age_num': building_age,
        'total_floors': total_floors,
        'floor': floor,
        'rooms': rooms,
        'halls': halls,
        'total_rooms': total_rooms,
        'is_new': 1 if building_age == 0 else 0,
        'floor_ratio': floor / total_floors if total_floors else np.nan,
        'size_per_room': size / total_rooms if total_rooms else np.nan,
        'sub_type': sub_type,
        'heating_type': heating_type,
        # редкие города и районы при обучении сложены в категорию «Другой»
        'city': city if city in package['top_cities'] else 'Другой',
        'county': county if county in package['top_counties'] else 'Другой',
    }

    one = pd.DataFrame([row])
    one = one[package['num_features'] + package['cat_features']]
    one = pd.get_dummies(one, columns=package['cat_features'], dtype=int)
    one = one.reindex(columns=package['columns'], fill_value=0)

    if package['needs_scaling']:
        one = pd.DataFrame(package['scaler'].transform(one),
                           columns=package['columns'])
    return one


def predict_price(**kwargs):
    features = make_features(**kwargs)
    log_price = package['model'].predict(features)[0]
    return float(np.exp(log_price))


def check_input(size, rooms, halls, total_floors, floor, building_age):
    """Возвращает список сообщений об ошибках. Пустой список = всё хорошо."""
    errors = []

    if size <= 0:
        errors.append('Площадь должна быть больше нуля.')
    if rooms + halls <= 0:
        errors.append('Укажите хотя бы одну комнату.')
    if floor > total_floors:
        errors.append('Этаж не может быть больше, чем всего этажей в доме '
                      '(указано: этаж ' + str(floor) + ', этажей ' + str(total_floors) + ').')
    if size / max(rooms + halls, 1) < 8:
        errors.append('Слишком маленькая площадь на комнату — проверьте площадь '
                      'и количество комнат.')
    if building_age > 100:
        errors.append('Возраст здания больше 100 лет — таких объектов в обучающих '
                      'данных не было, прогноз будет недостоверным.')
    return errors


# ----------------------------------------------------------------------
# Заголовок и вкладки
# ----------------------------------------------------------------------
st.title('🏠 Оценка стоимости недвижимости')
st.caption('Модель обучена на объявлениях площадки Zingat (Турция, 2018–2019). '
           'Цены — в турецких лирах (TRY).')

tab_predict, tab_dashboard, tab_help = st.tabs(
    ['📊 Прогноз цены', '📈 Дашборд', 'ℹ️ Справка'])


# ======================================================================
# ВКЛАДКА 1 — ФОРМА ПРОГНОЗА
# ======================================================================
with tab_predict:
    st.header('Введите характеристики объекта')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader('Основное')
        size = st.number_input('Площадь, м²', min_value=20, max_value=1000,
                               value=110, step=5,
                               help='Допустимый диапазон: от 20 до 1000 м²')
        rooms = st.number_input('Комнат', min_value=0, max_value=15, value=3, step=1)
        halls = st.number_input('Гостиных', min_value=0, max_value=5, value=1, step=1)
        st.caption('В турецких объявлениях это записывают как «' +
                   str(rooms) + '+' + str(halls) + '»')

    with col2:
        st.subheader('Дом')
        building_age = st.number_input('Возраст здания, лет', min_value=0,
                                       max_value=100, value=5, step=1,
                                       help='0 — новостройка')
        total_floors = st.number_input('Этажей в доме', min_value=1, max_value=40,
                                       value=8, step=1)
        floor = st.number_input('Этаж квартиры', min_value=-4, max_value=40,
                                value=4, step=1,
                                help='0 — первый/цокольный, отрицательные значения — подвал')

    with col3:
        st.subheader('Тип и расположение')
        sub_type = st.selectbox('Тип жилья', package['sub_types'],
                                index=list(package['sub_types']).index('Daire')
                                if 'Daire' in package['sub_types'] else 0)
        heating_type = st.selectbox('Отопление', package['heating_types'])

        cities = sorted([c for c in package['top_cities'] if c != 'Другой'])
        city = st.selectbox('Город', cities,
                            index=cities.index('İstanbul') if 'İstanbul' in cities else 0)

        # районы показываем только те, что реально встречаются в выбранном городе
        if data is not None:
            in_city = sorted(data[data['city'] == city]['county'].dropna().unique())
        else:
            in_city = sorted(package['top_counties'])
        if len(in_city) == 0:
            in_city = ['Другой']
        county = st.selectbox('Район', in_city)

    st.divider()

    if st.button('Рассчитать стоимость', type='primary', width='stretch'):
        errors = check_input(size, rooms, halls, total_floors, floor, building_age)

        if errors:
            for message in errors:
                st.error(message)
        else:
            try:
                price = predict_price(
                    size=size, rooms=rooms, halls=halls,
                    building_age=building_age, total_floors=total_floors,
                    floor=floor, sub_type=sub_type, heating_type=heating_type,
                    city=city, county=county)

                mae = package['mae']
                low = max(price - mae, 0)
                high = price + mae

                st.success('Прогноз готов')

                res1, res2, res3 = st.columns(3)
                res1.metric('Прогноз цены', f'{price:,.0f} TRY'.replace(',', ' '))
                res2.metric('Вероятный диапазон',
                            f'{low:,.0f} — {high:,.0f}'.replace(',', ' '))
                res3.metric('Цена за м²', f'{price / size:,.0f} TRY'.replace(',', ' '))

                mae_text = f'{mae:,.0f}'.replace(',', ' ')
                st.info('Диапазон построен как «прогноз ± средняя ошибка модели (MAE)». '
                        'Средняя ошибка модели на тестовых данных — '
                        f'{mae_text} TRY, или {package["mape"]:.1f}%.')

                if city not in package['top_cities']:
                    st.warning('Этот город не попал в тридцатку самых частых, '
                               'модель относит его к категории «Другой». '
                               'Прогноз будет менее точным.')

            except Exception as error:
                st.error('Не удалось рассчитать прогноз: ' + str(error))


# ======================================================================
# ВКЛАДКА 2 — ДАШБОРД
# ======================================================================
with tab_dashboard:
    if data is None:
        st.warning('Не найден файл app_data.csv. Запустите notebook целиком — '
                   'он создаёт этот файл на Этапе 4.')
    else:
        st.header('Статистика по датасету')

        st.sidebar.header('Фильтры дашборда')

        all_cities = sorted(data['city'].unique())
        picked_cities = st.sidebar.multiselect(
            'Города', all_cities,
            default=[c for c in ['İstanbul', 'Ankara', 'İzmir'] if c in all_cities])

        price_min, price_max = st.sidebar.slider(
            'Цена, млн TRY', 0.0, 10.0, (0.0, 3.0), step=0.1)

        picked_types = st.sidebar.multiselect(
            'Тип жилья', sorted(data['sub_type'].unique()))

        view = data.copy()
        if picked_cities:
            view = view[view['city'].isin(picked_cities)]
        if picked_types:
            view = view[view['sub_type'].isin(picked_types)]
        view = view[(view['price'] >= price_min * 1_000_000) &
                    (view['price'] <= price_max * 1_000_000)]

        if len(view) == 0:
            st.warning('Под выбранные фильтры не подошло ни одного объявления. '
                       'Смягчите условия слева.')
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric('Объявлений', f'{len(view):,}'.replace(',', ' '))
            k2.metric('Медианная цена',
                      f'{view["price"].median():,.0f}'.replace(',', ' '))
            k3.metric('Медианная площадь', f'{view["size"].median():.0f} м²')
            k4.metric('Цена за м²',
                      f'{(view["price"] / view["size"]).median():,.0f}'.replace(',', ' '))

            st.divider()

            c1, c2 = st.columns(2)

            with c1:
                fig = px.histogram(view, x='price', nbins=60,
                                   title='Распределение цены',
                                   labels={'price': 'Цена, TRY'})
                st.plotly_chart(fig, width='stretch')

            with c2:
                sample = view.sample(min(5000, len(view)), random_state=42)
                fig = px.scatter(sample, x='size', y='price', color='city',
                                 opacity=0.5, title='Площадь и цена',
                                 labels={'size': 'Площадь, м²', 'price': 'Цена, TRY'})
                st.plotly_chart(fig, width='stretch')

            c3, c4 = st.columns(2)

            with c3:
                by_city = (view.groupby('city')['price'].median()
                           .sort_values(ascending=False).head(15).reset_index())
                fig = px.bar(by_city, x='price', y='city', orientation='h',
                             title='Медианная цена по городам',
                             labels={'price': 'Медианная цена, TRY', 'city': ''})
                fig.update_yaxes(categoryorder='total ascending')
                st.plotly_chart(fig, width='stretch')

            with c4:
                popular = ['1+1', '2+1', '3+1', '4+1', '5+1']
                rooms_view = view[view['room_count'].isin(popular)]
                if len(rooms_view) > 0:
                    fig = px.box(rooms_view, x='room_count', y='price',
                                 category_orders={'room_count': popular},
                                 title='Цена по количеству комнат',
                                 labels={'room_count': 'Комнаты', 'price': 'Цена, TRY'})
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info('Под текущие фильтры не попали типовые планировки.')

            c5, c6 = st.columns(2)

            with c5:
                by_type = (view.groupby('sub_type')['price'].median()
                           .sort_values(ascending=False).reset_index())
                fig = px.bar(by_type, x='sub_type', y='price',
                             title='Медианная цена по типу жилья',
                             labels={'sub_type': '', 'price': 'Медианная цена, TRY'})
                st.plotly_chart(fig, width='stretch')

            with c6:
                by_age = view.groupby('building_age_num')['price'].median().reset_index()
                fig = px.line(by_age, x='building_age_num', y='price', markers=True,
                              title='Цена и возраст здания',
                              labels={'building_age_num': 'Возраст здания, лет',
                                      'price': 'Медианная цена, TRY'})
                st.plotly_chart(fig, width='stretch')

        st.divider()
        st.header('Сравнение моделей')

        if metrics is None:
            st.info('Файл app_metrics.csv не найден — запустите notebook целиком.')
        else:
            st.dataframe(metrics, width='stretch')

            plot_metrics = metrics.reset_index()
            plot_metrics.columns = ['Модель'] + list(plot_metrics.columns[1:])
            fig = px.bar(plot_metrics.sort_values('MAPE, %'),
                         x='MAPE, %', y='Модель', orientation='h',
                         title='Средняя ошибка в процентах (меньше — лучше)')
            fig.update_yaxes(categoryorder='total descending')
            st.plotly_chart(fig, width='stretch')


# ======================================================================
# ВКЛАДКА 3 — СПРАВКА
# ======================================================================
with tab_help:
    st.header('Справка')

    st.subheader('Что делает это приложение')
    st.write(
        'Приложение оценивает рыночную стоимость квартиры или дома по его '
        'характеристикам. Вы вводите площадь, количество комнат, этаж, возраст '
        'здания и расположение — модель машинного обучения возвращает ожидаемую '
        'цену и вероятный диапазон вокруг неё.')

    st.subheader('Как пользоваться')
    st.markdown(
        '1. Откройте вкладку **«Прогноз цены»**.\n'
        '2. Заполните все поля. Значения вне допустимых диапазонов ввести нельзя, '
        'а противоречивые сочетания приложение подсветит красным.\n'
        '3. Нажмите **«Рассчитать стоимость»**.\n'
        '4. Приложение покажет прогноз, диапазон и цену за квадратный метр.\n\n'
        'На вкладке **«Дашборд»** можно посмотреть, как устроены сами данные: '
        'фильтры находятся на панели слева.')

    st.subheader('Описание полей')
    fields = pd.DataFrame([
        ['Площадь, м²', 'Общая площадь объекта', 'от 20 до 1000'],
        ['Комнат', 'Число жилых комнат без гостиной', 'от 0 до 15'],
        ['Гостиных', 'Число гостиных. В Турции планировку пишут как «3+1»', 'от 0 до 5'],
        ['Возраст здания, лет', '0 означает новостройку', 'от 0 до 100'],
        ['Этажей в доме', 'Общее число этажей в здании', 'от 1 до 40'],
        ['Этаж квартиры', '0 — первый или цокольный, отрицательные — подвал', 'от -4 до 40'],
        ['Тип жилья', 'Daire — квартира, Villa — вилла, Müstakil Ev — частный дом', 'список'],
        ['Отопление', 'Тип системы отопления из объявления', 'список'],
        ['Город', '30 самых частых городов датасета', 'список'],
        ['Район', 'Районы выбранного города', 'список'],
    ], columns=['Поле', 'Что означает', 'Допустимые значения'])
    st.dataframe(fields, width='stretch', hide_index=True)

    st.subheader('О модели')
    n_train_text = f'{package["n_train"]:,}'.replace(',', ' ')
    st.markdown(
        f'**Модель:** {package["model_name"]}\n\n'
        f'**Обучающая выборка:** {n_train_text} объявлений\n\n'
        'Модель предсказывает логарифм цены, а результат переводится обратно в лиры. '
        'Это сделано потому, что цены распределены очень несимметрично.')

    m1, m2, m3 = st.columns(3)
    m1.metric('Средняя ошибка (MAE)',
              f'{package["mae"]:,.0f} TRY'.replace(',', ' '))
    m2.metric('Ошибка в процентах (MAPE)', f'{package["mape"]:.1f} %')
    m3.metric('R²', f'{package["r2"]:.2f}')

    st.caption('Метрики посчитаны на тестовой выборке, которая не использовалась '
               'ни при обучении, ни при подборе гиперпараметров.')

    st.subheader('Ограничения — прочитайте обязательно')
    st.warning(
        'Прогноз носит справочный характер и не является оценкой для сделки.')
    st.markdown(
        '* **Данные за 2018–2019 годы по Турции.** Для сегодняшних цен нужен '
        'пересчёт на инфляцию и изменение рынка.\n'
        '* **В данных нет части важных признаков:** состояния ремонта, наличия '
        'лифта и парковки, расстояния до центра и до метро. Модель их не видит, '
        'и это главная причина, почему точность ограничена.\n'
        '* **Средняя ошибка около 20%.** Для типовой квартиры прогноз будет '
        'достаточно точным, для редких и дорогих объектов — заметно хуже, '
        'потому что таких примеров в обучающих данных мало.\n'
        '* **Города вне списка** сводятся к категории «Другой», и точность по ним падает.\n'
        '* Модель обучена только на объявлениях **о продаже**. Для аренды она неприменима.')

    st.divider()
    st.subheader('О приложении')
    st.markdown(
        f'**Версия:** {APP_VERSION}\n\n'
        '**Автор:** укажите здесь своё ФИО и группу\n\n'
        '**Источник данных:** Zingat, файл `real_estate_data.csv`\n\n'
        '**Исходный код:** notebook `real_estate_price.ipynb` в этой же папке')
