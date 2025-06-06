import streamlit as st
from model_helper import predict

st.set_page_config(page_title="Animal Classifier", page_icon="🐾")

st.markdown("""
    <h1 style='text-align: center; color: #2e7d32;'>🦁 Animal Prediction 🐘</h1>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; background-color: #f0f2f6; padding: 20px; border-radius: 10px; color: black;'>
    <h4>This model can predict the following animals:</h4>
    🐕 Dog • 🐱 Cat • 🐎 Horse • 🐘 Elephant • 🦋 Butterfly<br>
    🐔 Chicken • 🐑 Sheep • 🕷️ Spider • 🐿️ Squirrel • 🐄 Cow
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png","webp"])

if uploaded_file:
    image_path = "temp_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        st.image(uploaded_file, caption="Uploaded File", use_container_width=True)
        prediction = predict(image_path)
        st.markdown(f"""
            <div style='background-color: #e8f4f9; 
                padding: 20px; 
                border-radius: 10px; 
                border-left: 6px solid #2196F3;
                margin: 10px 0px;'>
            <h3 style='color: #1976D2; margin: 0;'>
                🎯 Prediction Result: <span style='color: #2e7d32'>{prediction}</span>
            </h3>
            </div>
        """, unsafe_allow_html=True)
