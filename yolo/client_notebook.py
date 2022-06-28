# -*- coding: utf-8 -*-
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
#     display_name: py38-deepzoom
#     language: python
#     name: py38-deepzoom
# ---

# + [markdown] id="2F4hsFSsZXVK"
# # Notebook d'inference
#
# Ce notebook qui contient un script d'inférence est à transformer en webapp pour effectuer des prédictions.

# + [markdown] id="2n_BGU7LuIzt"
# ## Get an image

# + colab={"base_uri": "https://localhost:8080/", "height": 445} id="oO0CScw7ZXVe" outputId="86dccf64-dbb4-451d-865f-76a4ce677a22"
import base64
from io import BytesIO

import requests
from PIL import Image

response = requests.get(
    "https://unsplash.com/photos/YCPkW_r_6uA/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjU1NTU3ODcz&force=true&w=640"
)
img = Image.open(BytesIO(response.content))
if img.mode == "RGBA":
    img = img.convert("RGB")
# -

img

# + id="ge7GvT_bxeu0"
# Transform the image to base64 string

buffer = BytesIO()
img.save(buffer, format="JPEG")
img_str: str = base64.b64encode(buffer.getvalue()).decode("utf-8")

# + [markdown] id="GPV68jaquRpt"
# ## Set server URL

# + cellView="form" id="Nll0jZ-KuVRi"
# @title Set server URL
server_url = "https://wide-parks-rest-35-230-172-249.loca.lt"  # @param {type:"string"}

# + [markdown] id="1a-O_bsSujVd"
# ## Select model

# + cellView="form" id="XuI7JKJ6ulwx"
yolo_model = "yolov5m"  # @param ["yolov5s", "yolov5m", "yolov5l", "yolov5x"] {allow-input: true}

# + [markdown] id="7NX28T_Qum6f"
# ## Perform the inference

# + id="ZgmN3YcsupYU"
r = requests.post(f"{server_url}/predict", json={"model": yolo_model, "image": img_str})
if r.status_code == 200:
    result = r.json()
else:
    print(r.status_code)

# + id="2VZfAiPJAKPi"
# Let's get a list of all detected classes
classes = list(set(map(lambda x: x["class_name"], result["detections"])))

# + id="Tix7L_x9ZXVj"
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw, ImageFont


def draw_preds(image, preds, class_names):

    image = image.copy()

    colors = plt.cm.get_cmap("viridis", len(class_names)).colors
    colors = (colors[:, :3] * 255.0).astype(np.uint8)

    font = list(Path("arial.ttf").glob("**/*.ttf"))[0].name
    font = ImageFont.truetype(font=font, size=np.floor(3e-2 * image.size[1] + 0.5).astype("int32"))
    thickness = (image.size[0] + image.size[1]) // 300

    for pred in preds:
        score = pred["confidence"]
        predicted_class = pred["class_name"]

        label = "{} {:.2f}".format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top = pred["y_min"]
        left = pred["x_min"]
        bottom = pred["y_max"]
        right = pred["x_max"]
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for r in range(thickness):
            draw.rectangle(
                [left + r, top + r, right - r, bottom - r], outline=tuple(colors[class_names.index(predicted_class)])
            )
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=tuple(colors[class_names.index(predicted_class)]),
        )
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return image


# + colab={"base_uri": "https://localhost:8080/"} id="dlX7EOEGZXVl" outputId="c440b82b-d0de-4130-8f5c-c4ef00ca1561"
image_with_preds = draw_preds(img, result["detections"], classes)

# + colab={"base_uri": "https://localhost:8080/", "height": 445} id="uS3O8uBwZXVm" outputId="16221cfb-f5e4-4b29-bc37-940f7327a32b" pycharm={"name": "#%%\n"}
from IPython.display import display  # to display images

display(image_with_preds)
# -

image_with_preds
