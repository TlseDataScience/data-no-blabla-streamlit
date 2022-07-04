import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

st.title("My first dall-e application")


@st.cache(show_spinner=True)
def send_request(server_url, prompt, number_of_images):
    r = requests.post(
        f"{server_url}/dalle", json={"text": prompt, "num_images": number_of_images}
    )
    if r.status_code == 200:
        json = r.json()
        images = [Image.open(BytesIO(base64.b64decode(img))) for img in json]
    else:
        print(r.status_code)
        images = []

    return images


with st.form("Form"):
    st.text("Fill all data before clicking on SEND")

    server_url = st.text_input(label="server URL", value="https://myserver.tmp")
    prompt = st.text_input(label="Prompt")
    number_of_images = st.slider(
        label="Number of Images", min_value=1, max_value=12, value=3
    )

    submitted = st.form_submit_button(label="Send prompt to Dall-E")

if submitted:
    images = send_request(server_url, prompt, number_of_images)
    if len(images) > 0:
        st.balloons()

    cols = st.columns(3)

    for k, img in enumerate(images):
        col = cols[k % 3]

        with col:
            st.image(images[k])
