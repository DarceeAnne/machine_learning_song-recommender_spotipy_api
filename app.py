
import streamlit as st
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import pickle
import random
import os
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")



def main():

    #Initialize SpotiPy with user credentials
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(CLIENT_ID, CLIENT_SECRET))

    # Display the image in the sidebar
    image2= Image.open('darcee.png')
    st.sidebar.image(image2, use_column_width=True)
    st.write("")
    st.sidebar.title('App Created By:')
    st.sidebar.write('Darcee Caron')
    st.sidebar.write('[LinkedIn](https://www.linkedin.com/in/darceecarondataanalystpythonsqlpowerbi/)')
    image = Image.open('tape.png')
    st.sidebar.image(image, use_column_width=True)
    st.write("")
    st.sidebar.title('Special Thanks To:')
    st.sidebar.write('Guillaume')
    st.sidebar.write('Thomas')
    st.sidebar.write('Xavier')
    st.sidebar.write('Leo')
    st.sidebar.write('Martin')
    st.sidebar.write('Janine')
    st.sidebar.write('... for your input on what audio features to use for developing the machine learning model!')
    st.write("")
    st.sidebar.write('Andy')
    st.sidebar.write('Elnara')
    st.sidebar.write('... for being awesome teachers!')

    ##### Image
    image = Image.open('project_title_slide.png')
    st.image(image)


    st.write('Variety is the spice of life!')

    st.write("")

    st.write('With literally hundreds of millions of songs, there is a huge opportunity to discover new songs and artists that you will love. This song recommender allows you to provide the name of a song you like and will return a list of recommended songs for you to discover. Enjoy!')

    st.write("")

    st.write('Enjoy!')

    # Inserting multiple spaces
    st.markdown("---")
    for _ in range(1):
        st.markdown("")

    # Initialize user_song_confirmed
    user_song_confirmed = None

    # Ask the user to provide the name of a song they like
    user_song_title = st.text_input("What's the title of a song you love?")

    # Check if the user has entered a song title
    if user_song_title:
        # Song name sent to Spotipy, and Spotipy will return a list of songs
        results = sp.search(q=user_song_title, limit=3, market='FR')
        
        # Extract information from the search results
        if results['tracks']['items']:
            # Define a list to store the options for the dropdown selector
            song_options = []
            
            # Define a dictionary to map song names to their corresponding track IDs
            track_id_mapping = {}

            # Iterate through the search results and populate the options list with song and artist names
            for i, item in enumerate(results['tracks']['items'], start=1):
                artists = ", ".join(artist['name'] for artist in item['artists'])
                song_options.append(f"{item['name']} by {artists}")
                
                # Populate the track_id_mapping dictionary
                track_id_mapping[f"{item['name']} by {artists}"] = item['id']

            # User is asked to confirm which of the songs is the one they choose.
            user_song_confirmed = st.selectbox("Which is your song?", [""] + song_options)

            # Display the audio widget to play the selected song
            if user_song_confirmed:
                selected_track_id = track_id_mapping[user_song_confirmed]
                spotify_url = f"https://open.spotify.com/embed/track/{selected_track_id}"
                st.write(f"Listen to it on Spotify:")
                st.write(f'<iframe src="{spotify_url}" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', unsafe_allow_html=True)

    # Data input - Select a value
    st.write('How many songs would you like recommended?')
    requested_num_songs = st.slider('Select a value', min_value=1, max_value=10, value=5)

    # Check if user has confirmed a song
    if user_song_confirmed:
        # Get the track ID corresponding to the selected song
        selected_track_id = track_id_mapping[user_song_confirmed]

        # Track ID of the user's song sent to Spotipy and that song's audio features are returned.
        selected_track_features = sp.audio_features(selected_track_id)
        
        # Reshape selected song audio features
        track_feature_dict = selected_track_features[0]
        # Extract feature values from the dictionary
        feature_values = np.array([
            track_feature_dict['danceability'],
            track_feature_dict['speechiness'],
            track_feature_dict['acousticness'],
        ])
        # Reshape feature_values into a 2D array
        feature_values = feature_values.reshape(1, -1)
        
        # Load the MinMaxScaler object from the pickle file
        with open('scaler6.pickle', 'rb') as handle:
            scaler = pickle.load(handle)
        # Scale the feature values
        scaled_user_song_features = scaler.transform(feature_values)
        
        # Load the trained KMeans model from the pickle file
        with open('model6_km100.pickle', 'rb') as handle:
            kmeans100 = pickle.load(handle)
        # Predict the cluster of the selected song
        predicted_cluster = kmeans100.predict(scaled_user_song_features)
        
        # Read the CSV file into a DataFrame
        clustered_tracks_df_6 = pd.read_csv("tracks_clustered_df_6.csv")
        # Filter tracks in the same cluster as the predicted cluster value
        tracks_in_predicted_cluster = clustered_tracks_df_6[clustered_tracks_df_6['cluster_km100'] == predicted_cluster[0]]
        # Randomly select one track from the filtered tracks
        recommended_tracks = tracks_in_predicted_cluster.sample(n=requested_num_songs)
        
        # Display the recommended tracks with an embedded Spotify player for each track
        st.write(f"Recommended tracks from the predicted cluster based on '{user_song_confirmed}':")
        for index, row in recommended_tracks.iterrows():
            track_id = row['track_id']  # Use the correct column name for track ID
            spotify_url = f"https://open.spotify.com/embed/track/{track_id}"
            st.write(f"Listen to it on Spotify:")
            st.write(f'<iframe src="{spotify_url}" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', unsafe_allow_html=True)


    st.markdown("---")
    for _ in range(1):
        st.markdown("")

    st.header('About the Model')

    st.write('This Model was trained using K-Means clustering. Training data was extracted from Spotify using the Spotipy API. The traing data set included approximately 33K songs.')
    st.write('Spotify songs are giving attributes from a library of 11 audio features: acoutsticness, energy, danceability, instrumentalness,key, liveness, loudness, mode, speechiness, tempo, and valence.')
    st.write('Six different models were trained and testing using various combinations of these 11 audio features for clustering.')
    st.write('Model 1 included all 11 audio features and models 2-6 used three audio features each which were selected on the advice of some music-loving friends (credits in sidebar).')
    st.write('The final model used here is trained on three attributes: acousticness, danceability, and speechiness; and was trained on 600 clusters.')

    ##### Image
    image2 = Image.open('happy_listening.png')
    st.image(image2)

if __name__ == '__main__':
    main()