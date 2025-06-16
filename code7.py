from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import json
import os

model = SentenceTransformer("all-MiniLM-L6-v2")
if not os.path.exists("movie_embeddings.json"):
    # Load the model

    # Movie list with genres
    movies = [
        'dhoom3: thriller,horror,action,comedy,drama,romance',
        'baahubali: action,drama,historical,fantasy,adventure',
        'krrish: sci-fi,action,thriller,romance,drama',
        'singham: action,crime,thriller,drama',
        'andhadhun: thriller,mystery,crime,drama,comedy',
        'stree: horror,comedy,thriller,romance',
        'kabir singh: drama,romance,action',
        'bhool bhulaiyaa: horror,comedy,mystery,drama',
        'pathaan: action,thriller,spy,drama',
        'barfi: romance,drama,comedy,emotional',
        'queen: comedy,drama,romance,emotional',
        '3 idiots: comedy,drama,romance,emotional',
        'chakde! india: sports,drama,inspirational,emotional',
        'padmaavat: historical,drama,romance,action',
        'raazi: thriller,drama,espionage,romance',
    ]

    # Step 1: Generate and save embeddings to JSON
    movie_vectors = model.encode(movies)

    movie_data = []
    for movie, vector in zip(movies, movie_vectors):
        movie_data.append({
            "movie": movie,
            "embedding": vector.tolist()
        })

    with open("movie_embeddings.json", "w") as f:
        json.dump(movie_data, f, indent=2)

    print("✅ Embeddings saved to 'files/movie_embeddings.json'\n")

# Step 2: Load user input and compute similarity
user_input = input("🎬 Enter a movie name or genre description: ").lower()
query_vector = model.encode([user_input])

# Load embeddings from JSON
with open("files/movie_embeddings.json", "r") as f:
    loaded_data = json.load(f)

# Prepare data for comparison
movies_list = [item["movie"] for item in loaded_data]
vectors_list = np.array([item["embedding"] for item in loaded_data])

# Compute cosine similarity
similarities = cosine_similarity(query_vector, vectors_list)[0]

# Create and sort dataframe
df = pd.DataFrame({
    "movie": movies_list,
    "similarity": similarities
})

ranked_df = df.sort_values(by="similarity", ascending=False).reset_index(drop=True)

# Show top results
print("\n🎯 Top similar movies:\n")
print(ranked_df.head(10))
