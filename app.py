import streamlit as st
import requests

st.title("Распознавание хот-догов")

uploaded_file = st.file_uploader("Выберите картинку", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file)

    if st.button("Проверить"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post("http://127.0.0.1:8000/predict", files=files)
        data = response.json()
        st.write("Класс:", data["class_name"])
        st.write("Точность:", data["confidence"], "%")
