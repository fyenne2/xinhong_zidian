#
# !pip install -U face_recognition
# !pip install -U streamlit

import os
import face_recognition
import numpy as np
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
import re

try:
    os.mkdir("./upload_folder/")
except:
    pass

import streamlit as st


class XinhongZidian:
    def __init__(self):
        self.baseimg = "./base_imgs/base_suspect.jpg"
        self.target_path = "./base_imgs/"
        self.upload_folder = "./upload_folder/"
        self.uploaded_img = ""


def load_img(configs_):
    if configs_.uploaded_img == None:
        baseimg = configs_.baseimg
    else:
        baseimg = configs_.uploaded_img
    target_path = configs_.target_path
    pil_im = Image.open(baseimg)
    display(pil_im.resize((350, 350)))
    q = [i for i in os.listdir(target_path)]
    # This is an example of running face recognition on a single image
    # and drawing a box around each person that was identified.
    # Load a sample picture and learn how to recognize it.
    suspect_image = face_recognition.load_image_file(baseimg)
    suspect_face_encoding = face_recognition.face_encodings(suspect_image)[0]
    # # Load a second sample picture and learn how to recognize it.
    # biden_image = face_recognition.load_image_file("biden.jpg")
    # biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    # Create arrays of known face encodings and their names
    known_face_encodings = [
        suspect_face_encoding,
        # biden_face_encoding
    ]
    known_face_names = [
        "suspect",
        # "suspect2",
    ]
    print("Learned encoding for", len(known_face_encodings), "images.")
    st.image(pil_im, width=350)
    return known_face_encodings, known_face_names


def infer_(
    known_face_encodings, known_face_names, filepath
):  # Load an image with an unknown face
    unknown_image = face_recognition.load_image_file(filepath)
    # font = ImageFont.truetype(
    #     "/kaggle/input/facesreconization/DejaVuSans-Bold.ttf", size=24
    # )
    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)
    find_times_ = 0
    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        # print(face_distances)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        if face_distances < 0.25:
            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            # Draw a label with a name below the face
            bbox = draw.textbbox((0, 0), name)  # Provide a font if you have one
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.rectangle(
                ((left, bottom - text_height - 10), (right, bottom)),
                fill=(0, 0, 255),
                outline=(0, 0, 255),
            )
            draw.text(
                (left + 6, bottom - text_height - 5),
                name + "\n" + str(face_distances[0]),
                fill=(255, 255, 255, 255),
            )
            find_times_ += 1

    # Remove the drawing library from memory as per the Pillow docs
    del draw
    return find_times_, pil_image


if __name__ == "__main__":
    configs = XinhongZidian()
    st.write("# 1. upload your suspect img")
    configs.uploaded_img = st.file_uploader(
        "upload a img file", accept_multiple_files=False
    )
    if configs.uploaded_img == None:
        st.write("here is a default img")
    else:
        st.write(configs.uploaded_img)
    q = [
        i
        for i in os.listdir(configs.target_path)
        if i not in ["base_suspect.png", "base_suspect.jpg"]
    ]
    try:
        known_face_encodings, known_face_names = load_img(configs)
    except:
        st.write("---")
        st.warning("# face is not detected")

    st.write("---")
    st.write("# 2. after uploaded , hit run button to see results.")
    run_ = st.button("run")
    alltimes = 0

    if run_:
        my_bar = st.progress(0, text='running ...')
        for i, j in enumerate(q):
            my_bar.progress(i + 1, text=f'running {(str(i + 1))} `s image')
            find_times_, pil_image = infer_(
                known_face_encodings=known_face_encodings,
                known_face_names=known_face_names,
                filepath=configs.target_path + j,
            )
            if find_times_ != 0:
                st.image(pil_image)
                alltimes += 1

        if alltimes == 0:
            st.success("# not finding your suspect!")
