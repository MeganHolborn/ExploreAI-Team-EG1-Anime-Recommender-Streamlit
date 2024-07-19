"""

    Streamlit webserver application for generating user ratings based on historical preferences.

"""

import os

import joblib

# Data dependencies
import pandas as pd

# Streamlit dependencies
import streamlit as st
from PIL import Image

# Load your raw data
# raw = pd.read_csv("streamlit/train.csv")


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
        <p>Developed by Animex Analytics | Contact us at: info@animexanalytics.com</p>
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
        <p>Developed by Animex Analytics | Contact us at: info@animexanalytics.com</p>
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
        text = st.text_area("Enter User ID", "Type Here")

        # Create a anime title selection box
        anime_titles = ["Fullmetal Alchemist: Brotherhood", "Mushishi Zoku Shou"]
        st.selectbox("Select Anime/s", anime_titles)

        if st.button("Get Ratings"):

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
        <p>Developed by Animex Analytics | Contact us at: info@animexanalytics.com</p>
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
        <p>Developed by Animex Analytics | Contact us at: info@animexanalytics.com</p>
        </div>"""
        st.markdown("#")
        st.divider()
        st.markdown(footer_html, unsafe_allow_html=True)


# Required to let Streamlit instantiate our web app.
if __name__ == "__main__":
    main()
