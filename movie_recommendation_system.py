import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets on title
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop missing values
movies.dropna(inplace=True)

# Convert string representations of lists/dicts to actual lists/dicts
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Convert cast to a list of names, limiting to the first three
def convert3(obj):
    return [i['name'] for i in ast.literal_eval(obj)][:3]

movies['cast'] = movies['cast'].apply(convert3)

# Fetch director's name
def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['crew'] = movies['crew'].apply(fetch_director)

# Process overview
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Clean up spaces in genres, keywords, cast, and crew
for column in ['genres', 'keywords', 'cast', 'crew']:
    movies[column] = movies[column].apply(lambda x: [i.replace(" ", "") for i in x])

# Create tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Prepare the new DataFrame
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x)).str.lower()

# Vectorize the tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vector)

# Save data for recommendations
pickle.dump(new_df, open('movie_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

# Recommendation function
def recommend(movie):
    try:
        index = new_df[new_df['title'] == movie].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        print(f"Recommendations for '{movie}':")
        for i in distances[1:6]:
            print(new_df.iloc[i[0]].title)
    except IndexError:
        print("Movie not found!")

# Test the recommendation system
recommend('Avengers')
