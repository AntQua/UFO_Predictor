import streamlit as st

def add_image_styles():
    st.markdown(
        """
        <style>
        .custom-image {
            width: 600px;
            height: auto;
            border-radius: 2%;
            box-shadow: 0 4px 16px rgba(255, 255, 255, 0.8);
            margin: 20px auto; /* Center the image */
            display: block; /* Center the image */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
