from flask import Flask, render_template, request
import pickle
import pandas as pd
import requests
import random
import text2emotion as te

app = Flask(__name__)

# -----------------------
#  CONFIGURATION
# -----------------------
OMDB_API_KEY = "OMDb API"  

# Load ML-based dataset
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# -----------------------
#  EMOTION-BASED DATA
# -----------------------
movies_by_emotion = {
    'Happy': ['Zootopia', 'The Secret Life of Walter Mitty', 'Inside Out', 'Forrest Gump', 'La La Land'],
    'Sad': ['The Pursuit of Happyness', 'Marley & Me', 'A Walk to Remember', 'Hachi: A Dog\'s Tale', 'The Green Mile'],
    'Angry': ['John Wick', 'Gladiator', 'The Dark Knight', '300', 'The Equalizer'],
    'Fear': ['A Quiet Place', 'It', 'The Conjuring', 'Get Out', 'The Ring'],
    'Surprise': ['Inception', 'Now You See Me', 'The Prestige', 'Interstellar', 'Gone Girl']
}

# -----------------------
#  HELPER FUNCTIONS
# -----------------------
def predict_emotion(user_text):
    emotions = te.get_emotion(user_text)
    
   
    if not emotions or all(v == 0 for v in emotions.values()):
        return 'Surprise'
    
  
    filtered = {k: v for k, v in emotions.items() if v >= 0.2}
    
    if not filtered:
        return 'Surprise'
    
   
    dominant = max(filtered, key=filtered.get)
    return dominant.capitalize()


def get_movie_poster_omdb(movie_title):
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}"
    response = requests.get(url).json()
    return response.get("Poster", "https://via.placeholder.com/300x450?text=No+Image")

def recommend_emotion_movies(user_input):
    emotion = predict_emotion(user_input)
    movie_list = movies_by_emotion.get(emotion, [])
    recommendations = []
    for movie in random.sample(movie_list, len(movie_list)):
        poster = get_movie_poster_omdb(movie)
        recommendations.append({'title': movie, 'poster': poster})
    return emotion, recommendations

# -----------------------
#  DATASET (ML) RECOMMENDER
# -----------------------
def recommend_movie(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1]
    )[1:6]
    recommended_movies = []
    recommended_posters = []
    for i in distances:
        movie_title = movies.iloc[i[0]].title
        poster_url = get_movie_poster_omdb(movie_title)
        recommended_movies.append(movie_title)
        recommended_posters.append(poster_url)
    return recommended_movies, recommended_posters

# -----------------------
#  ROUTES
# -----------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/emotion', methods=['GET', 'POST'])
def emotion_page():
    emotion = None
    recommended_movies = []
    user_input = ""
    if request.method == 'POST':
        user_input = request.form['user_input']  # ✅ fixed key name
        emotion, recommended_movies = recommend_emotion_movies(user_input)
    return render_template('emotion.html', emotion=emotion, movies=recommended_movies, user_input=user_input)

@app.route('/similar', methods=['GET', 'POST'])
def similar_page():
    movie_list = movies['title'].values
    if request.method == 'POST':
        user_movie = request.form['movie']
        recommended_movies, recommended_posters = recommend_movie(user_movie)
        data = zip(recommended_movies, recommended_posters)
        return render_template('recommend.html', data=data, movie=user_movie)
    return render_template('index.html', movie_list=movie_list)

# -----------------------
#  MAIN
# -----------------------
if __name__ == '__main__':
    app.run(debug=True)
