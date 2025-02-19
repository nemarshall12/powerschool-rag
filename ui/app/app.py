import streamlit as st
import streamlit_shadcn_ui as ui
import requests
import logging

# Configure logging to display info messages in the console
logging.basicConfig(level=logging.INFO)

# Shoreland Lutheran High School logo URL
logo_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQdoxFVEHKAmx9DZIs349iizURnBZSlB6ZkUA&s"

st.set_page_config(page_title="Shoreland Q&A", page_icon=logo_url, layout="centered")

st.image(logo_url, width=100)
st.title("Shoreland Lutheran High School - PowerSchool Q&A Bot")

# Create a form so that pressing Enter submits it.
with st.form("query_form"):
    question = ui.input(
        "Ask your question here:",
        key="question_input",
        placeholder="Type your question here..."
    )
    submit_button = st.form_submit_button("Ask")

# Process the form submission
if submit_button:
    if question:
        try:
            logging.info("Sending API request for question: %s", question)
            response = requests.post(
                "http://api:8000/query",
                json={"query": question},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the response data as JSON
            data = response.json()
            logging.info("Received response: %s", data)

            # Display the formatted output
            st.subheader("Response:")
            st.markdown(data.get("answer", "No answer provided."))
        except requests.exceptions.RequestException as e:
            logging.error("API request error: %s", e)
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
