"""

    Streamlit webserver application for generating user ratings based on historical preferences.

"""

# Import dependencies
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Load data
anime_data = pd.read_csv(r"anime.csv")
user_ratings = pd.read_csv(r"train.csv")
util_matrix_norm = joblib.load(open(os.path.join(r"util_matrix_norm.pkl"), "rb"))
util_matrix_filtered = joblib.load(
    open(os.path.join(r"util_matrix_filtered.pkl"), "rb")
)

# Load models
model_knn = joblib.load(open(os.path.join(r"model_knn.pkl"), "rb"))


# Function to predict user ratings
def predict_rating(user_id, anime_id, sim_threshold=0.0):

    try:
        # Select the normalised review data for the user of interest
        user_data = util_matrix_norm.loc[user_id].to_numpy().reshape(1, -1)

        # Determine the indices of the similar users and their cosine distances from the user of interest
        sim_distances, sim_user_indices = model_knn.kneighbors(
            user_data, n_neighbors=21
        )

        # Calculate the similarity scores of the similar users (sim score = 1-distance), convert to a list and remove the first element
        sim_scores = (1 - sim_distances)[0].tolist()
        sim_scores.pop(0)

        # Convert the indices for similar users to a list and remove the first element
        sim_user_indices = sim_user_indices[0].tolist()
        sim_user_indices.pop(0)

        # Retrieve the ids of the similar users using the list of indices
        sim_user_ids = (
            util_matrix_norm.reset_index()
            .iloc[sim_user_indices]["user_id"]
            .values.tolist()
        )

        ratings = []
        weights = []

        # For every index, id in user_ids
        for i, sim_user_id in enumerate(sim_user_ids):

            # Get the similar user's rating for the anime of interest
            rating = util_matrix_filtered.loc[sim_user_id, anime_id]

            # Get the user's similarity score using the index value
            sim_score = sim_scores[i]

            # Check whether the user rating is valid and whether the user's similarity to the user of interest is above a defined threshold
            # If checks are passed, append weighted rating and similarity score to lists, else skip the user
            if not pd.isnull(rating) and sim_score > sim_threshold:
                ratings.append(rating * sim_score)
                weights.append(sim_score)

        try:
            # Calculate the predicted rating for the user of interest
            predicted_rating = sum(ratings) / sum(weights)

        except ZeroDivisionError:
            # If there are no valid ratings, return the average predicted rating given by all users
            predicted_rating = anime_data[anime_data["anime_id"] == anime_id][
                "rating"
            ].values[0]

    except KeyError:
        # If the user ID or anime ID was not present in the training data, return the average predicted given by all users
        predicted_rating = anime_data[anime_data["anime_id"] == anime_id][
            "rating"
        ].values[0]

    return predicted_rating


# The main function where we will build the actual app
def main():
    """Anime Recommender System"""

    # Creating sidebar with logo and navigation
    image = Image.open("Animex.png")
    st.sidebar.image(image)

    options = [
        "About",
        "Predict Your Ratings",
        "Your Recommendations",
        "Behind the Scenes",
    ]

    selection = st.sidebar.radio("Navigate To", options)

    # Building out the "About" page
    if selection == "About":

        # Create a banner

        # Add a title and banner
        st.image(
            "https://blog.playstation.com/tachyon/2016/10/unnamed-file-6.jpg",
            use_column_width=True,
        )

        st.title("Personalized Anime Recommender")
        st.divider()

        # Add information divided by subsections
        st.subheader(":grey[Introduction]")
        st.markdown("Some information here")

        st.subheader(":grey[App Features]")
        st.markdown("Some information here")

        st.subheader(":grey[How to Use the App]")
        st.markdown("Some information here")

        st.subheader(":grey[Meet the Team]")
        st.markdown("Some information here")

        # Add footer with contact information
        footer_html = """<div style='text-align: center;'>
        <p>Developed by Kawaii Consulting | Contact us at: info@kawaiiconsulting.com</p>
        </div>"""
        st.markdown("#")
        st.divider()
        st.markdown(footer_html, unsafe_allow_html=True)

    # Building out the "Get Recommendations" page
    if selection == "Your Recommendations":

        # Add a title and divider
        st.title("Your Recommendations")
        st.divider()

        # Add an information box
        st.info(
            "Get personalised recommendations based on your favourite shows and genres."
        )

        # Creating a text box for user input
        text = st.text_area("Enter User ID", "Type Here")

        # Create a genre selection box
        genres = ["Action", "Adventure"]
        st.selectbox("Select Genre", genres)

        # Create a number of recommendations selection box
        st.slider(
            "Number of Recommendations",
            1,
            30,
        )

        if st.button("Get Recommendations"):

            # Transforming user input with vectorizer
            test_cv = ""
            vect_text = test_cv.transform([text]).toarray()

            # Load your .pkl file with the model of your choice + make predictions
            predictor = joblib.load(
                open(os.path.join("streamlit/Logistic_regression.pkl"), "rb")
            )
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            st.success("Text Categorized as: {}".format(prediction))

        # Add footer with contact information
        footer_html = """<div style='text-align: center;'>
        <p>Developed by Kawaii Consulting | Contact us at: info@kawaiiconsulting.com</p>
        </div>"""
        st.markdown("#")
        st.divider()
        st.markdown(footer_html, unsafe_allow_html=True)

    # Building out the "Predicted Ratings" page
    if selection == "Predict Your Ratings":

        # Add a title and divider
        st.title("Predict Your Ratings")
        st.divider()

        # Add an information box
        st.info(
            "Predict what rating you would give an anime you haven't watched before."
        )

        # Creating a text box for user input
        user_id = st.number_input(
            "Enter your User ID. If invalid, the overall average rating will be returned.",
            value=None,
            placeholder="User ID",
            step=1,
        )

        # Create a anime title selection box

        # Get a list of all animes
        anime_ids = anime_data["anime_id"].unique()

        # Select anime that the user has not rated
        rated_anime_ids = user_ratings[user_ratings["user_id"] == user_id][
            "anime_id"
        ].unique()

        unrated_anime = anime_data[~anime_data["anime_id"].isin(rated_anime_ids)][
            "name"
        ].unique()

        selected_anime = st.selectbox("Select an Anime", unrated_anime)

        # Get the user's predicted rating for the selected anime
        if st.button("Predict Your Ratings"):

            selected_anime_id = anime_data[anime_data["name"] == str(selected_anime)][
                "anime_id"
            ].values[0]

            prediction = predict_rating(user_id, selected_anime_id)

            st.write("Predicted Rating:", prediction)

        # Add footer with contact information
        footer_html = """<div style='text-align: center;'>
        <p>Developed by Kawaii Consulting | Contact us at: info@kawaiiconsulting.com</p>
        </div>"""
        st.markdown("#")
        st.divider()
        st.markdown(footer_html, unsafe_allow_html=True)

    # Building out the "Behind the Scenes" page
    if selection == "Behind the Scenes":

        # Add a title
        st.title("Behind the Scenes")
        st.divider()

        # Add information divided by subsections
        st.subheader(":grey[Project Methods]")
        st.markdown("Some information here")

        st.subheader(":grey[Future Work]")
        st.markdown("Some information here")

        # Add footer with contact information
        footer_html = """<div style='text-align: center;'>
        <p>Developed by Kawaii Consulting | Contact us at: info@kawaiiconsulting.com</p>
        </div>"""
        st.markdown("#")
        st.divider()
        st.markdown(footer_html, unsafe_allow_html=True)


# Required to let Streamlit instantiate our web app.
if __name__ == "__main__":
    main()
