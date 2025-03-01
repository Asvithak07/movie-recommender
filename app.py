import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("netflix1.csv")  # Ensure this file is uploaded in Colab

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["listed_in"])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create movie index mapping
movie_indices = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()

# Function to get recommendations
def recommend_movies(title, num_recommendations=5):
    title = title.lower()  # Convert input to lowercase
    if title not in movie_indices:
        return ["Movie not found in dataset."]

    idx = movie_indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices_list = [i[0] for i in sim_scores]

    return df["title"].iloc[movie_indices_list].tolist()

# Streamlit UI
st.title("🎬 Movie Recommender System")
st.write("Enter a movie name to get recommendations.")

# User input
movie_name = st.text_input("Enter movie name:")

if st.button("Get Recommendations"):
    if movie_name:
        recommendations = recommend_movies(movie_name)
        st.write("### Recommended Movies:")
        for movie in recommendations:
            st.write(f"✅ {movie}")
    else:
        st.write("❌ Please enter a valid movie name.")
