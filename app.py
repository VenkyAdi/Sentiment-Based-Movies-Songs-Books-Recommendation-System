import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from scipy.special import softmax
import pickle


@st.cache_resource
def load_model_and_tokenizer():
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return model, tokenizer


@st.cache_resource
def load_datasets():
    with open('books_dataset.pkl', 'rb') as books_file:
        books = pickle.load(books_file)

    with open('movies_dataset.pkl', 'rb') as movies_file:
        movies = pickle.load(movies_file)

    with open('songs_dataset.pkl', 'rb') as songs_file:
        songs = pickle.load(songs_file)

    return books, movies, songs


model, tokenizer = load_model_and_tokenizer()
books, movies, songs = load_datasets()

labels = ['Negative', 'Neutral', 'Positive']

def analyze_sentiment(text):
    tweet_words = [word if not word.startswith('@') else '@user' for word in text.split()]
    tweet_proc = " ".join(tweet_words)

    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    sentiment_probabilities = {labels[i]: scores[i] for i in range(len(labels))}
    return sentiment_probabilities

def extract_genres(text):
    cleaned_text = ' '.join([word for word in text.split() if not word.startswith('@')]).lower()

    keyword_lst = {
            'Crime': ['murder', 'detective', 'robbery', 'criminal', 'crime', 'investigation', 'mystery', 'thief', 'homicide', 'heist', 'police', 'law', 'suspect', 'evidence', 'forensic', 'courtroom', 'gang', 'underworld', 'corruption', 'witness', 'interrogation', 'fugitive', 'justice', 'prison', 'conspiracy', 'hostage', 'ransom', 'bribery', 'blackmail', 'organized crime', 'alibi', 'surveillance', 'fear', 'tension', 'distrust'],
            
            'Comedy': ['funny', 'humor', 'laugh', 'comedy', 'joke', 'hilarious', 'stand-up', 'satire', 'parody', 'slapstick', 'witty', 'gag', 'prank', 'skit', 'absurd', 'farce', 'silly', 'laughable', 'banter', 'punchline', 'caricature', 'clown', 'spoof', 'sarcasm', 'one-liner', 'comedic timing', 'laugh track', 'joy', 'playfulness', 'lighthearted'],
                    
            'Family': ['family', 'kids', 'children', 'parent', 'siblings', 'wholesome', 'bonding', 'generations', 'togetherness', 'values', 'babysitting', 'holiday', 'relatives', 'tradition', 'care', 'protection', 'support', 'warmth', 'nurturing'],
                    
            'Fantasy': ['magic', 'fantasy', 'supernatural', 'dragons', 'wizards', 'mythical', 'spells', 'enchantment', 'sorcery', 'creatures', 'medieval', 'prophecy', 'portal', 'parallel worlds', 'sword', 'epic', 'gods', 'fairy tale', 'otherworldly', 'ancient magic', 'awe', 'wonder', 'imagination', 'hope', 'desire'],
                    
            'Romance': ['love', 'romance', 'relationship', 'passion', 'affection', 'courtship', 'heartbreak', 'chemistry', 'soulmate', 'intimacy', 'flirting', 'proposal', 'jealousy', 'date', 'romantic', 'first love', 'longing', 'secret admirer', 'wedding', 'honeymoon', 'desire', 'empathy', 'trust', 'connection', 'vulnerability'],
                    
            'Horror': ['scary', 'horror', 'fear', 'ghost', 'monster', 'terror', 'creepy', 'haunted', 'psychological', 'gore', 'jump scare', 'nightmare', 'serial killer', 'demonic', 'possessed', 'curse', 'creature', 'slasher', 'undead', 'zombie', 'darkness', 'blood', 'evil', 'stalker', 'scream', 'panic', 'dread', 'unease', 'shock'],
                    
            'Action': ['action', 'fight', 'war', 'explosion', 'combat', 'chase', 'rescue', 'battle', 'mission', 'hero', 'assassin', 'showdown', 'weapon', 'gunfight', 'adrenaline', 'martial arts', 'special forces', 'escape', 'survival', 'spy', 'enemy', 'revolution', 'duel', 'vigilante', 'courage', 'fearlessness', 'bravery', 'determination', 'strength'],
                    
            'Sci_Fi': ['sci-fi', 'aliens', 'space', 'future', 'robot', 'technology', 'time travel', 'cyberpunk', 'extraterrestrial', 'spaceship', 'galaxy', 'parallel universe', 'dystopia', 'clones', 'virtual reality', 'artificial intelligence', 'cyborg', 'android', 'terraforming', 'wormhole', 'curiosity', 'discovery', 'inspiration', 'isolation'],
                    
            'Drama': ['drama', 'emotional', 'life', 'struggle', 'relationship', 'family', 'conflict', 'personal growth', 'tragedy', 'heartbreak', 'sacrifice', 'redemption', 'betrayal', 'self-discovery', 'crisis', 'moral dilemma', 'grief', 'injustice', 'reconciliation', 'forgiveness', 'despair', 'hope', 'resilience', 'sympathy', 'compassion'],
                    
            'Adventure': ['adventure', 'explore', 'journey', 'quest', 'discovery', 'expedition', 'danger', 'wilderness', 'treasure', 'survival', 'challenge', 'exploration', 'uncharted', 'island', 'daring', 'heroic', 'wild', 'legend', 'artifact', 'map', 'compass', 'excitement', 'curiosity', 'thrill', 'boldness', 'anticipation'],
                    
            'Music': ['music', 'singing', 'band', 'concert', 'song', 'performance', 'lyrics', 'guitar', 'piano', 'melody', 'festival', 'orchestra', 'musical', 'album', 'recording', 'rhythm', 'harmony', 'composer', 'soundtrack', 'hit single', 'tune', 'note', 'emotion', 'joy', 'creativity', 'passion', 'inspiration'],

            'Fiction': ['storytelling', 'characters', 'narrative', 'conflict', 'identity', 'loneliness', 'despair', 'triumph', 'loss', 'hope', 'brokenness', 'resilience', 'emptiness', 'growth', 'emotional conflict'],

            'Detective and Mystery': ['detective', 'mystery', 'suspense', 'isolation', 'fear', 'paranoia', 'puzzle', 'anxiety', 'revelation', 'tension', 'secrets', 'justice', 'uncertainty', 'doubt', 'desperation'],

            'Christian Life': ['faith', 'struggle', 'forgiveness', 'redemption', 'inner peace', 'confusion', 'hope', 'guilt', 'empathy', 'spiritual conflict', 'forgiveness', 'compassion', 'sacrifice', 'healing'],

            'Adventure': ['journey', 'exploration', 'loneliness', 'danger', 'excitement', 'isolation', 'risk', 'fear', 'survival', 'uncertainty', 'inner strength', 'bravery', 'wilderness', 'unknown', 'endurance'],

            'American Fiction': ['individualism', 'struggle', 'freedom', 'identity', 'alienation', 'loss', 'hope', 'conflict', 'despair', 'rebuilding', 'brokenness', 'overcoming adversity', 'hope', 'disillusionment', 'dream'],

            'Fantasy Fiction': ['magic', 'supernatural', 'myth', 'epic', 'quest', 'inner struggle', 'destiny', 'sacrifice', 'mystery', 'conflict', 'mythical beings', 'escape', 'prophecy', 'hope', 'overcoming darkness'],

            'Science Fiction': ['technology', 'future', 'isolation', 'alienation', 'progress', 'paranoia', 'dystopia', 'utopia', 'fear', 'identity', 'existential crisis', 'innovation', 'struggle', 'hope', 'artificial intelligence'],

            'Juvenile Fiction': ['innocence', 'friendship', 'adventure', 'growing up', 'emotional conflict', 'family', 'learning', 'hope', 'loneliness', 'belonging', 'self-discovery', 'fear', 'imagination', 'exploration', 'wholesome values'],

            'Historical Fiction': ['history', 'struggle', 'sacrifice', 'identity', 'survival', 'loss', 'emotional depth', 'alienation', 'hope', 'triumph', 'cultural conflict', 'change', 'brokenness', 'resilience', 'renewal'],

            'Drama': ['relationships', 'inner turmoil', 'heartbreak', 'loneliness', 'grief', 'emotional conflict', 'redemption', 'self-discovery', 'betrayal', 'loss', 'resilience', 'healing', 'brokenness', 'hope', 'reconciliation'],

            'Country Life': ['nature', 'simplicity', 'community', 'loneliness', 'values', 'family bonds', 'conflict', 'peace', 'solitude', 'hardships', 'emotional growth', 'connection to the land', 'isolation', 'endurance', 'healing'],

            'Arthurian Romances': ['chivalry', 'honor', 'epic quests', 'betrayal', 'sacrifice', 'inner conflict', 'loneliness', 'heroism', 'destiny', 'legend', 'tragedy', 'nobility', 'forgiveness', 'hope'],

            'Dysfunctional Families': ['conflict', 'alienation', 'emotional struggle', 'loneliness', 'resentment', 'healing', 'grief', 'brokenness', 'communication issues', 'guilt', 'forgiveness', 'trauma', 'inner turmoil', 'reconciliation', 'hope'],

            'Christmas Stories': ['family', 'togetherness', 'joy', 'loss', 'hope', 'emotional warmth', 'loneliness', 'compassion', 'healing', 'celebration', 'generosity', 'grief', 'connection', 'miracles', 'tradition'],

            'Human Cloning': ['identity', 'existential crisis', 'alienation', 'technology', 'fear of the unknown', 'inner conflict', 'loneliness', 'struggle for meaning', 'moral dilemmas', 'isolation', 'self-awareness', 'loss of self', 'hope', 'humanity', 'consciousness'],

            'Literary Collections': ['reflections', 'human experience', 'emotional depth', 'existentialism', 'grief', 'loneliness', 'hope', 'growth', 'self-discovery', 'resilience', 'fear', 'loss', 'joy', 'brokenness', 'rebuilding'],
            
            'Sex': ['sex','fuck','adult','desire', 'intimacy', 'passion', 'lust', 'physical attraction', 'romantic connection', 'sensuality', 'seduction', 'vulnerability', 'consent', 'taboo', 'infidelity', 'heartbreak', 'sexual tension', 'emotional connection', 'sexual identity', 'eroticism', 'temptation', 'jealousy', 'power dynamics', 'shame', 'guilt', 'pleasure', 'fantasy', 'regret', 'obsession', 'trust', 'betrayal', 'exploration', 'boundaries'],
            
            'Bollywood': ['hindi', 'bollywood', 'indian cinema', 'hindi movies', 'bollywood films'],

            'Hindi Songs': ['hindi', 'bollywood', 'hindustani', 'desi', 'hindipop', 'hindimusic'],
            
            'Punjabi Songs': ['punjabi', 'bhangra', 'punjabimusic'],
            
            'Tamil Songs': ['tamil', 'kollywood', 'tamilmusic'],

    }

    matched_genres = [genre for genre, keywords in keyword_lst.items() if any(keyword in cleaned_text for keyword in keywords)]
    return matched_genres


GOOGLE_BOOKS_API_KEY = 'AIzaSyD2zbn0m2l1fGZhW1wmmqNYgDiEdHFMNsI' 
OMDB_API_KEY = '7e2b1671' 


SPOTIFY_CLIENT_ID = '08ccb6a47f80405c8ba8b6617e446523'
SPOTIFY_CLIENT_SECRET = '2b3b4a3ef3124e4b9f5087485b7e52d9'
spotify_credentials = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=spotify_credentials)

def fetch_book_poster(title):
    url = f"https://www.googleapis.com/books/v1/volumes?q={title}&key={GOOGLE_BOOKS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data['totalItems'] > 0:
        return data['items'][0]['volumeInfo'].get('imageLinks', {}).get('thumbnail', None)
    return None

def fetch_movie_poster(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data['Response'] == 'True':
        return data.get('Poster', None)
    return None

def fetch_song_poster(song_title, artist_name):
    search_query = f"track:{song_title} artist:{artist_name}"
    result = sp.search(q=search_query, type='track', limit=1)

    if result['tracks']['items']:
        song = result['tracks']['items'][0]
        return song['album']['images'][0]['url'] if song['album']['images'] else None
    return None

def recommend_books(genre):
    recommended_books = books[books['categories'].str.contains('|'.join(genre), case=False, na=False)].head(10)
    book_recommendations = []

    for _, row in recommended_books.iterrows():
        title = row['title']
        poster_url = fetch_book_poster(title)
        book_recommendations.append({'title': title, 'poster_url': poster_url})

    return book_recommendations

def recommend_movies(genre):
    recommended_movies = movies[movies['tags'].str.contains('|'.join(genre), case=False)].head(10)
    movie_recommendations = []

    for _, row in recommended_movies.iterrows():
        title = row['title']
        poster_url = fetch_movie_poster(title)
        movie_recommendations.append({'title': title, 'poster_url': poster_url})

    return movie_recommendations

def recommend_songs(genre):
    genre_songs = songs[songs['text'].str.contains('|'.join(genre), case=False)].head(10)
    song_recommendations = []

    for _, row in genre_songs.iterrows():
        song_title = row['song']
        artist = row['artist']
        poster_url = fetch_song_poster(song_title, artist)
        song_recommendations.append({'song': song_title, 'artist': artist, 'poster_url': poster_url})

    return song_recommendations


st.title("MoodMatch – Aligns with mood-based and genre preferences")

user_input = st.text_input("FeelVibes – Input your emotions, get content by genre ['Movies - Books - Songs']", "")

recommended_movies = []
recommended_books = []
recommended_songs = []

if user_input:
    
    sentiment_probs = analyze_sentiment(user_input)
    max_sentiment = max(sentiment_probs, key=sentiment_probs.get)

    
    # st.subheader("Sentiment Analysis:")
    # for label, probability in sentiment_probs.items():
    #     st.write(f"{label}: {probability:.4f}")
    # st.write(f"Max sentiment: **{max_sentiment}**")

    
    extracted_genres = extract_genres(user_input)
    st.subheader("Extracted Genres:")
    st.write(", ".join(extracted_genres) if extracted_genres else "No genres detected")

    
    st.subheader("Recommendations:")
    
    if extracted_genres:
        recommended_movies = recommend_movies(extracted_genres)
        recommended_books = recommend_books(extracted_genres)
        recommended_songs = recommend_songs(extracted_genres)

        
    if recommended_movies:
        st.write("**Movies:**")

        movies_per_row = 5

        # Calculate number of rows needed
        num_rows = (len(recommended_movies) + movies_per_row - 1) // movies_per_row

        for row in range(num_rows):
            
            cols = st.columns(movies_per_row)
            for col in range(movies_per_row):
                idx = row * movies_per_row + col
                if idx < len(recommended_movies):
                    movie = recommended_movies[idx]
                    with cols[col]:
                        # Check if the poster_url exists
                        if movie['poster_url']:
                            # Check if genre is available, if not display 'No genre found'
                            genre = movie.get('genre', 'No genre found')
                            
                            st.markdown(f"""
                            <div style="display: flex; flex-direction: column; align-items: center; margin: 10px;">
                                <img src="{movie['poster_url']}" width="150" style="margin-bottom: 10px;">
                                <div><strong>{movie['title']}</strong></div>
                                <div style="font-size: 12px;">{genre}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.write("No poster available")
                else:
                    # Display if no title found for the slot
                    with cols[col]:
                        st.write("")



        
    if recommended_books:
        st.write("**Books:**")

        
        books_per_row = 5

        
        num_rows = (len(recommended_books) + books_per_row - 1) // books_per_row  

        for row in range(num_rows):
            
            cols = st.columns(books_per_row)
            for col in range(books_per_row):
                idx = row * books_per_row + col
                if idx < len(recommended_books):
                    book = recommended_books[idx]
                    with cols[col]:
                        if book['poster_url']:  
                            
                            st.markdown(f"""
                            <div style="display: flex; flex-direction: column; align-items: center; margin: 10px;">
                                <img src="{book['poster_url']}" width="150" style="margin-bottom: 10px;">
                                <div>{book['title']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.write(f"Image not available for {book['title']}")  
                else:
                    
                    with cols[col]:
                        st.write("")  


            
    if recommended_songs:
        st.write("**Songs:**")

        
        songs_per_row = 5

     
        num_rows = (len(recommended_songs) + songs_per_row - 1) // songs_per_row

        for row in range(num_rows):
            
            cols = st.columns(songs_per_row * 2 - 1)  

            for col in range(songs_per_row):
                idx = row * songs_per_row + col
                col_idx = col * 2  

                if idx < len(recommended_songs):
                    song = recommended_songs[idx]
                    with cols[col_idx]:
                        if song['poster_url']:
                            st.image(song['poster_url'], width=150)
                        else:
                            st.write("No Image Available")
                        
                        st.write(f"{song['song']} - {song['artist']}")
                    

                if col < songs_per_row - 1:
                    cols[col_idx + 1].empty()  
                else:
                    st.write("")
    else:
        st.write("No genres detected. Please provide more specific input ['I love music drama fantasy'].")

