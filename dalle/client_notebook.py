# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + cellView="form" id="JwQId5scclAg"
# @title Set server URL
server_url = "https://brave-months-share-34-86-217-132.loca.lt"  # @param {type:"string"}

# + [markdown] id="i4aXez6-fZcW"
# # Select a prompt

# + cellView="form" id="ptVwIcDxfYlF"
prompt = "a donkey wearing sunglasses and a hat"  # @param ["a donkey wearing sunglasses and a hat", "antique photo of a clown riding a T-Rex", "a teddy bear on a skateboard on the moon", "a giant rubber duck on a lake"] {allow-input: true}

# + id="4IeoqW6eg7mx"
import base64
from io import BytesIO

import requests
from PIL import Image

# + id="8ZP9CwJtg-N5"
r = requests.post(f"{server_url}/dalle", json={"text": prompt, "num_images": 6})
if r.status_code == 200:
    json = r.json()
    images = [Image.open(BytesIO(base64.b64decode(img))) for img in json]
else:
    print(r.status_code)

# + [markdown] id="IL9tYui6oup0"
# # Display the result

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="-RBCYIM-kCHh" outputId="2b1e307d-7a73-40ac-d12b-92d0cf9e635d"
for image in images:
    display(image)

# + id="vP90rqumobs2"

