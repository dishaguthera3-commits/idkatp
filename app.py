import streamlit as st
import pandas as pd
import joblib
import requests
import os

# --------------------------
# Page Setup
# --------------------------
st.set_page_config(
    page_title="AI Health Chatbot",
    page_icon="💬",
    layout="wide"
)
st.title("💬 AI Health & Diet Chatbot (SDG 3)")
st.markdown("Talk to the bot to get **BMI calculation**, **diet recommendations**, and **activity plans** based on your profile.")

# --------------------------
# Download ML files if not present
# --------------------------
def download_file(url, local_path):
    if not os.path.exists(local_path):
        r = requests.get(url)
        with open(local_path, "wb") as f:
            f.write(r.content)

# Replace these URLs with your GitHub raw links
urls = {
    "diet_model.pkl": "https://raw.githubusercontent.com/USERNAME/REPO/main/models/diet_model.pkl",
    "activity_model.pkl": "https://raw.githubusercontent.com/USERNAME/REPO/main/models/activity_model.pkl",
    "le_activity.pkl": "https://raw.githubusercontent.com/USERNAME/REPO/main/models/le_activity.pkl",
    "le_diet.pkl": "https://raw.githubusercontent.com/USERNAME/REPO/main/models/le_diet.pkl",
    "le_rec_diet.pkl": "https://raw.githubusercontent.com/USERNAME/REPO/main/models/le_rec_diet.pkl",
    "le_rec_activity.pkl": "https://raw.githubusercontent.com/USERNAME/REPO/main/models/le_rec_activity.pkl"
}

for fname, url in urls.items():
    download_file(url, fname)

# --------------------------
# Load Models & Encoders
# --------------------------
diet_model = joblib.load("diet_model.pkl")
activity_model = joblib.load("activity_model.pkl")
le_activity = joblib.load("le_activity.pkl")
le_diet = joblib.load("le_diet.pkl")
le_rec_diet = joblib.load("le_rec_diet.pkl")
le_rec_activity = joblib.load("le_rec_activity.pkl")

# --------------------------
# Chatbot Interface
# --------------------------
st.subheader("Enter your details below:")

age = st.number_input("Age", 10, 100, 25)
height = st.number_input("Height (cm)", 100, 220, 170)
weight = st.number_input("Weight (kg)", 30, 150, 65)
activity = st.selectbox("Activity Level", ["Low", "Moderate", "High"])
diet = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan", "Mixed"])
sleep = st.slider("Average Sleep Hours", 3, 12, 7)

if st.button("Get Recommendations"):

    # Calculate BMI
    bmi = round(weight / ((height/100)**2), 2)
    if bmi < 18.5:
        category = "Underweight"
        color = "blue"
    elif bmi < 25:
        category = "Normal weight"
        color = "green"
    elif bmi < 30:
        category = "Overweight"
        color = "orange"
    else:
        category = "Obese"
        color = "red"

    st.markdown(f"<h3 style='color:{color}'>Your BMI: {bmi} ({category})</h3>", unsafe_allow_html=True)

    # Encode categorical features
    activity_enc = le_activity.transform([activity])[0]
    diet_enc = le_diet.transform([diet])[0]

    input_df = pd.DataFrame([[age, bmi, activity_enc, sleep, diet_enc]],
                            columns=['Age','BMI','ActivityLevel_enc','SleepHours','DietType_enc'])

    # ML Predictions
    pred_diet_enc = diet_model.predict(input_df)[0]
    pred_diet = le_rec_diet.inverse_transform([pred_diet_enc])[0]

    pred_activity_enc = activity_model.predict(input_df)[0]
    pred_activity = le_rec_activity.inverse_transform([pred_activity_enc])[0]

    # Display recommendations
    st.subheader("💡 Personalized Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Diet Plan**")
        st.info(pred_diet)
    with col2:
        st.markdown("**Activity Plan**")
        st.success(pred_activity)

st.markdown("---")
st.caption("Developed as a college AI project for SDG 3: Good Health & Well-Being")