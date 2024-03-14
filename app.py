import os

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

model_path = 'model.h5'
loaded_model = load_model(model_path)

#if os.path.exists(model_path):
#    loaded_model = load_model(model_path)
#    st.success("Model loaded successfully.")
#else:
#   st.error(f"File not found at: {model_path}")

# Fungsi untuk pra-pemrosesan teks
def predict_cyber_bullying(new_teks, loaded_model):
    max_len = loaded_model.input_shape[1]

    # Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([new_teks])
    sekuens_teks = tokenizer.texts_to_sequences([new_teks])
    padding_teks = pad_sequences(sekuens_teks, maxlen=max_len)

    # Melakukan prediksi
    prediction = loaded_model.predict(padding_teks)

    if prediction[0][0] > 0.5:
        sentiment = 'Sentiment Positif'
    else:
        sentiment = 'Sentiment Negatif !!!'
    persen = round(prediction[0][0] * 100)

    result = {
        'teks': new_teks,
        'prediksi': f'{persen}% - {sentiment}'
    }

    return result


st.title(f"Natural Language Processing")
image_url = 'https://satelitweb.com/wp-content/uploads/2023/12/apa-itu-NLP.jpg'
st.image(image_url, use_column_width=True)
st.subheader("Sentiment Analytics Text Classification")
with st.expander("Penjelasan Singkat"):
    st.write("""
        NLP (Natural Language Processing) adalah cabang kecerdasan buatan yang fokus pada interaksi 
        antara komputer dan bahasa manusia. Tujuannya adalah memungkinkan komputer memahami, menginterpretasi, 
        dan merespons teks atau ucapan manusia secara efektif. Aplikasinya meliputi pemrosesan bahasa alami, 
        terjemahan bahasa, analisis sentimen, dan lainnya.
        """)
st.markdown('---')
st.subheader("Test Model NLP")
# Menambahkan catatan atau informasi tambahan
st.info('Masukan Kalimat (Bahasa Indonesia) untuk pengetesan model Deep Learning')
# Minta input dari pengguna
user_input = st.text_area('Masukan Kalimat', '')
if st.button('Prediksi'):
    prediction_result = predict_cyber_bullying(user_input, loaded_model)
    st.write('Prediksi :', prediction_result['prediksi'])
    st.caption("Terkadang prediksi model tidak 100% akurat")

st.markdown('---')
st.caption('© 2024 Wildan Mujjahid Robbani - All Rights Reserved')