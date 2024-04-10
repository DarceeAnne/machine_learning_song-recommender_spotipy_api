# Machine Learning Song Recommender Spotipy API

**<p style="font-size:24px;">[Check Out the App Here](https://machine-learning-song-recommender.streamlit.app/)</p>**


## Data Overview
For this project, a dataset comprising 34,000 pre-gathered songs and an additional 14,000 songs obtained via API calls on playlists were provided. These songs were sourced from various platforms, including Spotify. Each song in the Spotify dataset includes 11 audio features, including:
- Acousticness
- Danceability
- Energy
- Instrumentalness
- Key
- Liveness
- Loudness
- Mode
- Speechiness
- Tempo
- Valence
- Model Development

## Model Development

To create an effective music recommendation model, six different models were tested. The first model utilized all 11 audio features, while the remaining five models were derived from crowd-sourced opinions from music enthusiasts. These enthusiasts were asked to choose the top three audio features they believed would be the best predictors of similar songs.

Here are their picks:

![Crowd Source Image](https://raw.githubusercontent.com/DarceeAnne/machine_learning_song-recommender_spotipy_api/main/crowd%20source.png)


Each of the six models was trained and evaluated using a set of five songs. The efficacy of each model was scored on a 5-point scale based on the similarity of the recommended songs to the input songs. 

![Evaluation Image](https://raw.githubusercontent.com/DarceeAnne/machine_learning_song-recommender_spotipy_api/main/eval.png)

Ultimately, Model 6 emerged as the most effective. This final model was trained on three attributes: acousticness, danceability, and speechiness, using K-Means clustering with 600 clusters.

![Thanks Image](https://raw.githubusercontent.com/DarceeAnne/machine_learning_song-recommender_spotipy_api/main/thanks.png)
