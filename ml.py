import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# %config InlineBackend.figure_format = 'retina'

import warnings

warnings.filterwarnings("ignore")


def get_audio_features_artist_top_tracks(artist_list):
    """
    This function takes a list of artist names as input and returns a dictionary of average audio features
    for each artist's top tracks.

    Args:
        artist_list (list): A list of strings representing the names of the artists to retrieve audio features for.

    Returns:
        A dictionary where the keys are artist names (strings) and the values are numpy arrays of average audio features
        for that artist's top tracks.
    """

    # Create a dictionary to store the audio features for each artist
    artist_audio_features = {}

    # Loop over each artist in the input list
    for artist_name in artist_list:
        # Use time.sleep to avoid hitting the Spotify API rate limit
        time.sleep(0.1)

        # Use the Spotify search API to find the artist's URI
        results = sp.search(q=artist_name, type="artist")
        artist_uri = results["artists"]["items"][0]["uri"]

        # Use the Spotify artist_top_tracks API to get the artist's top tracks
        top_tracks = sp.artist_top_tracks(artist_uri)["tracks"]

        # Get the URIs for each track in the top tracks
        track_uris = [track["uri"] for track in top_tracks]

        # Split the track URIs into batches of 50 and get the audio features for each batch
        for i in range(0, len(track_uris), 50):
            batch = track_uris[i : i + 50]
            # Append the audio features for the batch to the list of audio features for the artist
            artist_audio_features.setdefault(artist_name, []).extend(
                sp.audio_features(batch)
            )

    # Create a new dictionary to store the average audio features for each artist
    artist_audio_feats = {}

    # Define the audio feature names to use in the output dictionary
    keys = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "valence",
        "time_signature",
        "duration_ms",
        "tempo",
    ]

    # Loop over each artist in the artist_audio_features dictionary
    for artist_name, track_features in artist_audio_features.items():
        # Create a list to store the audio features for each track
        artist_track_feats = []
        # Loop over each track's audio features
        for track_feat in track_features:
            # Append the audio features to the list for the current track
            artist_track_feats.append([track_feat.get(key) for key in keys])
        # Calculate the mean audio features across all tracks for the artist
        artist_audio_feats[artist_name] = np.mean(np.array(artist_track_feats), axis=0)

    return artist_audio_feats


def get_audio_features_artist_all_tracks(artist_list):
    """
    Given a list of artist names, retrieves audio features for all their tracks and returns a dictionary with the
    artist names as keys and their audio features as values.

    Args:
        artist_list (list): List of artist names (strings)

    Returns:
        dict: Dictionary with artist names (strings) as keys and their audio features as values (numpy arrays)
    """

    artist_audio_features = (
        {}
    )  # Initialize an empty dictionary to store audio features for each artist

    # Loop through each artist in the list and retrieve their top tracks
    for artist_name in artist_list:
        time.sleep(0.1)  # To avoid hitting the API rate limit
        results = sp.search(q=artist_name, type="artist")
        artist_uri = results["artists"]["items"][0]["uri"]
        albums = sp.artist_albums(artist_uri, album_type="album")
        album_uris = [album["uri"] for album in albums["items"]]
        tracks = []
        for album_uri in album_uris:
            album_tracks = sp.album_tracks(album_uri)["items"]
            tracks += album_tracks

        # Retrieve the URIs for each track and store them in a list
        track_uris = [track["uri"] for track in tracks]

        # Retrieve audio features for each batch of 50 tracks and store them in the artist_audio_features dictionary
        for i in range(0, len(track_uris), 50):
            batch = track_uris[i : i + 50]
            artist_audio_features[artist_name] = sp.audio_features(batch)

    artist_audio_feats = (
        {}
    )  # Initialize an empty dictionary to store the mean audio features for each artist's tracks

    # Define a list of keys to retrieve from the audio features dictionaries
    keys = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "valence",
        "time_signature",
        "duration_ms",
        "tempo",
    ]

    # Loop through each artist in the artist_audio_features dictionary and calculate the mean audio features for their tracks
    for artist_name, track_features in artist_audio_features.items():
        artist_track_feats = (
            []
        )  # Initialize an empty list to store the audio features for each track
        for track_feat in track_features:
            artist_track_feats.append(
                [track_feat.get(key) for key in keys]
            )  # Append the specified keys from the audio features dictionary to the artist_track_feats list
            artist_audio_feats[artist_name] = np.mean(
                np.array(artist_track_feats), axis=0
            )  # Calculate the mean audio features for the artist's tracks and store them in the artist_audio_feats dictionary

    return artist_audio_feats  # Return the dictionary with artist names as keys and their mean audio features as values


def get_cosim_artist_df(favorite_artists, df, n):
    """
    Given a list of favorite_artists, a dataframe containing audio features for a set of artists, and an integer n,
    returns a dictionary containing the top n artists most similar to each of the favorite_artists according to cosine
    similarity on their audio feature vectors.

    Args:
        favorite_artists (list): a list of favorite artists
        df (pandas.DataFrame): a dataframe containing audio features for a set of artists
        n (int): number of most similar artists to return for each favorite artist

    Returns:
        artists_scores (dict): a dictionary with favorite_artists as keys and dataframes as values, where each dataframe
        contains the top n most similar artists to the corresponding favorite artist according to cosine similarity on
        their audio feature vectors.
    """

    # Standardize the audio features using StandardScaler
    ss = StandardScaler()
    df_scaled = ss.fit_transform(df)
    df = pd.DataFrame(data=df_scaled, index=df.index)

    artists_scores = {}

    # For each favorite artist, compute cosine similarity scores with all other artists and store top n artists
    for artist_name in favorite_artists:
        artist_array = np.array(df.T[artist_name]).reshape(1, -1)
        dataset_array = df.drop(index=artist_name).values
        cosim_scores = cosine_similarity(artist_array, dataset_array).flatten()

        artist_names_array = df.drop(index=artist_name).index.values

        df_result = pd.DataFrame(
            data={
                "artist": artist_names_array,
                "cosim_" + artist_name: cosim_scores,
            }
        )

        # Remove favorite artists from the resulting dataframe
        df_result = remove_artists(df_result, favorite_artists)

        # Sort the dataframe in descending order by cosine similarity scores and return top n artists
        df_result = df_result.sort_values(
            by="cosim_" + artist_name, ascending=False
        ).head(n)
        artists_scores[artist_name] = df_result.reset_index(drop=True)

    return artists_scores


def get_artist_images(artist_list, image_dict={}):
    """
    Retrieves the image URLs for a list of artists from the Spotify API.

    Args:
    - artist_list: a list of strings representing the names of the artists to retrieve images for
    - image_dict: an optional dictionary with pre-existing artist image URLs to update

    Returns:
    A dictionary with artist names as keys and image URLs as values.
    """
    # Check if any artists already have images in the image_dict and remove them from artist_list if so
    artist_list = list(set(artist_list) - set(image_dict.keys()))

    for artist_name in artist_list:
        time.sleep(0.1)

        results = sp.search(q=artist_name, type="artist")

        if results["artists"]["items"]:
            image_dict[artist_name] = results["artists"]["items"][0]["images"][0]["url"]

    return image_dict

def get_single_artist_image(artist_name):
    """
    Retrieves the image URLs for a list of artists from the Spotify API.
    
    Args:
    - artist_list: a list of strings representing the names of the artists to retrieve images for
    - image_dict: an optional dictionary with pre-existing artist image URLs to update
    
    Returns:
    A dictionary with artist names as keys and image URLs as values.
    """
    # Check if any artists already have images in the image_dict and remove them from artist_list if so
    time.sleep(0.01)

    results = sp.search(q=artist_name, type='artist')

    if results['artists']['items']:
        image_url = results['artists']['items'][0]['images'][0]['url']
    else:
        image_url = None
            
    return image_url

def get_artist_spotify_url(artist_name):
    '''
    Retrieves the Spotify URL for a list of artists from the Spotify API.
    '''
    results = sp.search(q=artist_name, type='artist')
    if results['artists']['items']:
        return results['artists']['items'][0]['external_urls']['spotify']
    else:
        return None


def plot_artist_ranking(df_similarity_scores_from_artist, image_dict):
    """
    Plots a bar chart of the most similar artists to the given artist along with their images.

    Args:
    - df_similarity_scores_from_artist (pd.DataFrame): DataFrame with similarity scores for each artist.
    - image_dict (dict): Dictionary containing artist name to image URL mappings.

    Returns:
    - None
    """
    import warnings

    warnings.filterwarnings("ignore")

    # Get the main artist's image
    main_artist = image_dict[df_similarity_scores_from_artist.columns.values[1]]
    response = requests.get(main_artist)
    img = Image.open(BytesIO(response.content))

    # Create the plot
    fig, axs = plt.subplots(
        1, len(df_similarity_scores_from_artist) + 1, figsize=(50, 50)
    )
    axs[0].imshow(img)
    axs[0].axis("off")
    axs[0].set_title(
        "COACHELLA\nHere are the artist\nmost similar to\n{}".format(
            df_similarity_scores_from_artist.columns.values[1]
        ),
        fontname="Luminari",
        fontsize=50,
        fontstyle="italic",
    )

    # Set the color scheme
    color = "#e3dac9"
    axs[0].set_facecolor(color=color)
    fig.patch.set_facecolor(color)

    # Add each artist's image to the plot
    for i, row in df_similarity_scores_from_artist.iterrows():
        response = requests.get(image_dict[row["artist"]])
        img = Image.open(BytesIO(response.content))
        axs[i + 1].imshow(img)
        axs[i + 1].axis("off")
        axs[i + 1].set_title(
            f"{i+1}. {row['artist']}\nSimilarity: {row[df_similarity_scores_from_artist.columns.values[1]]*100:.2f} %",
            fontname="Luminari",
            fontsize=30,
            fontstyle="italic",
        )

    # Show the plot
    plt.show()


def remove_artists(df, artists_to_remove):
    """
    Remove rows from a pandas DataFrame that correspond to given artists.

    Args:
        df (pandas.DataFrame): DataFrame to remove rows from.
        artists_to_remove (list): List of artist names to remove from the DataFrame.

    Returns:
        pandas.DataFrame: Modified DataFrame with specified artists removed.
    """

    # Iterate through list of artists to remove
    for artist in artists_to_remove:
        # If artist not found in DataFrame, print message and continue iteration
        if artist not in df["artist"].values:
            # print(f"Artist {artist} not found in dataframe.")
            continue
        # Remove rows corresponding to given artist from DataFrame
        df = df[df["artist"] != artist]

    return df


def get_artists_from_playlist(playlist_uri):
    """
    Returns a list of unique artist names from a given playlist.

    Args:
    playlist_uri (str): The Spotify URI of the playlist.

    Returns:
    list: A list of unique artist names from the playlist.
    """

    playlist_tracks = sp.playlist_tracks(playlist_uri)["items"]
    playlist_artist = [
        track["track"]["artists"][0]["name"] for track in playlist_tracks
    ]

    return list(set(playlist_artist))


def create_audio_features_df(artist_audio_features_dict):
    """
    Create a pandas DataFrame with the audio features for a group of artists.

    Args:
    artist_audio_features_dict (dict): A dictionary where each key is an artist name and each value is a dictionary of audio features.

    Returns:
    pd.DataFrame: A pandas DataFrame with the artist names as index and the audio features as columns.
    """

    artists = []
    audio_vec = []

    for key, val in artist_audio_features_dict.items():
        artists.append(key)
        audio_vec.append(val)

    audio_vec = np.array(audio_vec)
    keys = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "valence",
        "time_signature",
        "duration_ms",
        "tempo",
    ]

    return pd.DataFrame(audio_vec, index=artists, columns=keys)


def save_dict_to_file(my_dict, filename):
    # save the dictionary to a file
    with open(filename, "wb") as f:
        pickle.dump(my_dict, f)


def load_dict_from_file(filename):
    # load the dictionary from the file
    with open(filename, "rb") as f:
        my_dict = pickle.load(f)
    return my_dict


def get_top_artist_recommendations(df_user, df_fest, n):
    """
    Returns a dictionary containing the top n recommended artists for each user based on their listening history and festival lineup.

    Parameters:
    -----------
    df_user : pandas dataframe
        A dataframe containing the user's listening history with columns representing audio features and rows representing tracks.
    df_fest : pandas dataframe
        A dataframe containing the festival lineup with columns representing audio features and rows representing artists.
    n : int
        The number of recommended artists to return for each user.

    Returns:
    --------
    artist_scores : dict
        A dictionary containing the top n recommended artists for each user.
        Keys are the names of users and values are pandas dataframes with columns 'artist' and 'cosine_similarity'.
    """
    # Concatenate the user and festival dataframes
    df = pd.concat([df_user, df_fest])

    # Scale the dataframe using StandardScaler
    ss = StandardScaler()
    df_scaled = ss.fit_transform(df)

    # Define the column names
    keys = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "valence",
        "time_signature",
        "duration_ms",
        "tempo",
    ]

    # Create a new dataframe with the scaled values and column names
    df = pd.DataFrame(data=df_scaled, index=df.index, columns=keys)

    # Get the user and festival dataframes
    df_user = df.iloc[0 : len(df_user)].copy()
    df_fest = df.iloc[len(df_user) :].copy()

    # Convert the dataframes to arrays
    user_array = df_user.values
    fest_array = df_fest.values

    # Calculate the cosine similarity scores
    cosim_scores = cosine_similarity(user_array, fest_array)

    # Create a new dataframe with the cosine similarity scores and artist names
    df_result = pd.DataFrame(
        data=cosim_scores, columns=df_fest.index.values, index=df_user.index.values
    )

    m = []
    for i in df_fest.index.values:
        if i in df_user.index.values:
            m.append(i)
    df_result = df_result.drop(columns = m)

    # Create a dictionary to store the top recommended artists for each user
    artist_scores = {}
    for artist in df_result.index.values:
        artist_scores[artist] = (
            df_result.T[artist]
            .sort_values(ascending=False)
            .head(n)
            .reset_index()
            .rename(columns={"index": "artist"})
        )

    return artist_scores


def plot_all_recommended_artists(artist_scores, artist_images):
    """
    Plots images of recommended artists based on their cosine similarity scores in a grid of subplots.

    Parameters:
    -----------
    artist_cosim_scores : dict
        A dictionary containing cosine similarity scores of recommended artists.
        Keys are the names of users and values are pandas dataframes with columns 'artist' and 'cosine_similarity'.

    artist_images : dict
        A dictionary containing URLs of artist images. Keys are the names of artists and values are the URLs as strings.

    Returns:
    --------
    None
    """
    recomended_artist = []
    for _, scores in artist_scores.items():
        for i, artist in scores.iterrows():
            recomended_artist.append(artist["artist"])
    rec_artist = sorted(list(set(recomended_artist)))

    # Compute the number of rows and columns based on the length of rec_artist
    num_images = len(rec_artist)
    num_cols = 5
    num_rows = (
        num_images + num_cols - 1
    ) // num_cols  # round up to the nearest integer

    plt.switch_backend("Agg")
    # Create a figure with subplots for each artist
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20))

    # Loop through each artist and plot their image and ranking
    for i, artist in enumerate(rec_artist):
        row = i // num_cols  # integer division to get the row index
        col = i % num_cols  # modulo operator to get the column index

        # Load the image from the URL
        response = requests.get(artist_images[artist])
        img = Image.open(BytesIO(response.content))

        # Plot the image
        axs[row, col].imshow(img)
        axs[row, col].axis("off")

        # Add the artist name and ranking as a title
        axs[row, col].set_title(
            f"{artist}", fontname="Luminari", fontsize=10, fontstyle="italic"
        )
        color = "#e3dac9"
        axs[row, col].set_facecolor(color=color)
        fig.patch.set_facecolor(color)

        # Adjust the spacing between the subplots
        # fig.subplots_adjust(left=0.005, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1)

    # Show the plot
    plt.savefig("static/images/recommended_artists.png", bbox_inches="tight")


client_id = os.environ.get("CLIENT_ID")
client_secret = os.environ.get("CLIENT_SECRET")

# remove the .cache file so a new one can be generated

try:
    os.remove(".cache")
except OSError:
    pass

auth_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret
)

sp = spotipy.Spotify(auth_manager=auth_manager)


def ml(playlist_uri):
    favorite_artists = get_artists_from_playlist(playlist_uri=playlist_uri)
    user_artists_audio_feats = get_audio_features_artist_top_tracks(favorite_artists)
    # user_artists_audio_feats = load_dict_from_file('static/pkl/user.pickle')
    # artist_images_dict = load_dict_from_file("static/pkl/artist_images.pickle")
    fest_artists_audio_feats = load_dict_from_file("static/pkl/fest.pickle")
    # artist_images = get_artist_images(favorite_artists, artist_images_dict)
    df_fest = create_audio_features_df(fest_artists_audio_feats)
    df_user = create_audio_features_df(user_artists_audio_feats)
    artist_scores = get_top_artist_recommendations(df_user, df_fest, 3)
    # plot_all_recommended_artists(artist_scores, artist_images)

    du = {}
    for artist in artist_scores.keys():
        tdict = {}
        du[artist] = {}
        du[artist]["image"] = get_single_artist_image(artist)
        du[artist]["spotify"] = get_artist_spotify_url(artist)
        for i in range(len(artist_scores[artist])):
            tdict[artist_scores[artist]["artist"][i]] = {
                "image": get_single_artist_image(artist_scores[artist].iloc[i]['artist']),
                "score": artist_scores[artist][artist][i],
                "spotify": get_artist_spotify_url(artist_scores[artist].iloc[i]['artist'])
            }
        du[artist]["recommendations"] = tdict
    # print(du)

    with open("static/data/dict.json") as test_file:
        duh = json.load(test_file)
    duh[playlist_uri] = du
    with open("static/data/dict.json", "w") as outfile:
        json.dump(duh, outfile)
    
    return du
