"""

    Simple Streamlit webserver application for serving developed recommender
	models.

"""

import os

import joblib

# Data dependencies
import pandas as pd

# Streamlit dependencies
import streamlit as st

# Load your raw data
# raw = pd.read_csv("streamlit/train.csv")


# The main function where we will build the actual app
def main():
    """Anime Recommender System"""

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Recommend Anime")
    st.subheader("Your Next Binge-Watch Awaits")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Recommend Anime", "About"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "About" page
    if selection == "About":
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

    # Building out the "Recommend Anime" page
    if selection == "Recommend Anime":
        st.info(
            "Get personalised recommendations based on your favourite shows and genres."
        )

        # # Creating a text box for user input
        # text = st.text_area("Enter Text", "Type Here")

        # Create a anime title selection box
        anime_titles = ["Fullmetal Alchemist: Brotherhood", "Mushishi Zoku Shou"]
        st.selectbox("Select an anime", anime_titles)

        # Create a genre selection box
        genres = ["Action", "Adventure"]
        st.selectbox("Select a genre", genres)

        if st.button("Recommend"):

            # Transforming user input with vectorizer
            test_cv = ""
            vect_text = test_cv.transform([text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(
                open(os.path.join("streamlit/Logistic_regression.pkl"), "rb")
            )
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))


# Required to let Streamlit instantiate our web app.
if __name__ == "__main__":
    main()
