import os
import uuid
import requests
import json

from ml import ml  # Importing custom ml module

from flask import request
from flask import Flask, render_template, redirect, url_for, session
from dotenv import load_dotenv
from flask_oauthlib.client import OAuth

# Loading environment variables from .env file
load_dotenv()

app = Flask(
    __name__, template_folder="templates"
)  # Creating Flask application instance
app.secret_key = os.environ.get("SECRET_KEY")  # Replace with your secret key

# Set up OAuth for Spotify
oauth = OAuth(app)
spotify = oauth.remote_app(
    "spotify",
    consumer_key=os.environ.get("CLIENT_ID"),
    consumer_secret=os.environ.get("CLIENT_SECRET"),
    base_url="https://api.spotify.com/v1/",
    request_token_url=None,
    access_token_url="https://accounts.spotify.com/api/token",
    authorize_url="https://accounts.spotify.com/authorize",
)


# Routes
@app.route("/")
def home():
    if "spotify_token" in session:
        # Render home page if authenticated
        return render_template("home.html")
    else:
        # Redirect to login page if not authenticated
        return render_template("login.html")


@app.route("/name", methods=["POST"])
def name():
    playlist_uri = request.form[
        "playlist_uri"
    ]  # Extracting playlist uri from POST request
    # du = ml(playlist_uri)  # Calling ml function from ml module, passing playlist uri as argument
    filename = os.path.join(app.static_folder, 'data', 'dict.json')
    with open(filename) as test_file:
        duh = json.load(test_file)
    if playlist_uri in duh.keys():
        du = duh[playlist_uri]
        print("oneee")
    else:
        du = ml(playlist_uri)
        print("twooo")
    return render_template(
        "name.html", docs = du
    )  # Rendering name.html with playlist uri


@app.route("/login")
def login():
    # Generate a unique state value
    state = str(uuid.uuid4())
    # Store the state value in the session
    session["state"] = state
    return spotify.authorize(
        callback=url_for("authorized", _external=True), state=state
    )


@app.route("/logout")
def logout():
    session.pop("spotify_token", None)
    return redirect(url_for("home"))


@app.route("/authorized")
def authorized():
    print("Authorizing...")
    resp = spotify.authorized_response()
    if resp is None:
        # Redirect to login page if authorization failed
        return redirect(url_for("login"))
    # Verify the state value to ensure it's not a forgery attack
    if session.get("state") != request.args.get("state"):
        return "Invalid state parameter", 401
    # Remove the state value from the session
    session.pop("state", None)
    print("access_token: ", resp["access_token"])
    session["spotify_token"] = (resp["access_token"], "")

    # Redirect to home page if authorization successful
    access_token = session["spotify_token"][0]
    headers = {"Authorization": f"Bearer {access_token}"}

    response = requests.get("https://api.spotify.com/v1/me/playlists", headers=headers)

    if response.status_code == 200:
        playlists = []
        for playlist in response.json()["items"]:
            playlists.append({"name": playlist["name"], "uri": playlist["uri"]})
        print(playlists)
        return render_template("home.html", data=playlists)
    else:
        print("Error:", response.status_code, response.text)


# Decorator function that returns the Spotify access token stored in the user's session
@spotify.tokengetter
def get_spotify_oauth_token():
    return session.get("spotify_token")


# Run the Flask application
if __name__ == "__main__":
    # Start the Flask development server on localhost:5555, with debug mode set by the DEBUG environment variable
    app.run(host="0.0.0.0", port=5555, debug=os.environ.get("DEBUG"))
