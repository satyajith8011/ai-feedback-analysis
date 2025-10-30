import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Feedback AI Demo", layout="wide")
st.title("AI Customer Feedback Analysis (Demo)")

st.markdown("Upload the `feedback_cleaned.csv` exported from Colab. The demo model was trained on simulated labels for demonstration purposes.")

uploaded = st.file_uploader("Upload feedback CSV", type=['csv'])
default_csv_local = "feedback_cleaned.csv"

def load_model():
    possible_paths = ['app/sentiment_model.pkl', 'sentiment_model.pkl', '/app/sentiment_model.pkl']
    for p in possible_paths:
        if os.path.exists(p):
            try:
                return pickle.load(open(p, 'rb'))
            except Exception:
                pass
    return None

model = load_model()

if uploaded is None:
    if os.path.exists(default_csv_local):
        if st.button("Load sample feedback_cleaned.csv from repo"):
            df = pd.read_csv(default_csv_local)
        else:
            df = None
    else:
        df = None
else:
    df = pd.read_csv(uploaded)

if df is None:
    st.info("Upload a CSV file that contains a 'feedback' column OR click 'Load sample...' if available.")
else:
    st.subheader("Preview")
    st.dataframe(df.head(10))

    if model is None:
        st.warning("Demo model (sentiment_model.pkl) not found in repo. Please upload it to the repo root or app folder.")
    else:
        if 'feedback' not in df.columns:
            st.error("CSV must include a column named 'feedback'.")
        else:
            texts = df['feedback'].astype(str).tolist()
            preds = model.predict(texts)
            label_map = {0:'Negative', 1:'Neutral', 2:'Positive'}
            df['sentiment'] = [label_map.get(p, 'Unknown') for p in preds]

            st.subheader("Sentiment distribution")
            st.bar_chart(df['sentiment'].value_counts())

            st.subheader("Samples with sentiment")
            st.dataframe(df[['feedback','sentiment']].head(50))

            st.subheader("Download results")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV with predictions", csv, "feedback_with_sentiment.csv", "text/csv")
