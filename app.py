import streamlit as st
import pickle

# Load saved model
@st.cache_resource
def load_model():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

model, vectorizer = load_model()

# UI
st.title("💊 Pharmacogenomic Risk Prediction System")
st.write("Enter a drug review to predict adverse reaction risk.")

user_input = st.text_area("Drug Review")

if st.button("Predict"):
    if user_input.strip() != "":
        review_vector = vectorizer.transform([user_input])
        prediction = model.predict(review_vector)[0]
        probability = model.predict_proba(review_vector)[0]

        if prediction == 1:
            st.error("⚠️ High Risk Detected")
            st.write(f"Confidence: {round(probability[1]*100,2)}%")
        else:
            st.success("✅ Low Risk")
            st.write(f"Confidence: {round(probability[0]*100,2)}%")
    else:
        st.warning("Please enter a review.")