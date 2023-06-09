import streamlit as st
import keras
import glob
import face_recognition
import cv2
import os
import numpy as np
import tensorflow as tf
model=model = keras.models.load_model('femalevsmale_mobilenetv2_ft_80f')
st.set_page_config(page_title="PH", layout="wide")
st.markdown("<h1 style='text-align: center; color: grey;'>Gender Classification</h1>", unsafe_allow_html=True)


def detect_faces(image, output_folder):
    # Use face_recognition library for face detection
    face_locations = face_recognition.face_locations(image)

    # Check if any faces are detected
    if len(face_locations) > 0:
        # Loop over the face locations
        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Extract the face ROI (Region of Interest) without the green square
            face_roi = image[top:bottom, left:right]

            # Generate a unique filename for saving the face image
            filename = f"face_{i}.jpg"
            output_path = os.path.join(output_folder, filename)

            # Save the face image without the green square
            cv2.imwrite(output_path, face_roi)

            # Draw a green square around the face on the original image
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        # Convert BGR image to RGB for displaying with plt
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        col1, col2, col3 = st.columns(3)
        with col2:
            st.image(image_rgb, caption="Face(s) detected")

        IMAGE_SIZE = [160, 160]
        class_names = ["Female", "Male"]
        test_dataset = tf.keras.utils.image_dataset_from_directory("test",
                                                                    batch_size=20,
                                                                    image_size=IMAGE_SIZE,
                                                                    shuffle=False)
        # Retrieve a batch of images from the test set
        image_batch, label_batch = test_dataset.as_numpy_iterator().next()
        predictions = model.predict_on_batch(image_batch).flatten()

        # Apply a sigmoid since our model returns logits
        predictions = tf.nn.sigmoid(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1)

        image_files = glob.glob("test/ex/*")
        num_images = len(image_files)
        st.write("--------------------------------------------")
        for i in range(num_images):
            co1, co2, co3 = st.columns(3)
            with co2:
                st.image(image_batch[i].astype("uint8"), caption=class_names[predictions[i]])
        for image in image_files:
            os.remove(image)
    else:
        st.write("No faces detected.")


# st.set_page_config(page_title="PH",page_icon=":tada:")
file = st.file_uploader("Please choose an Image")
# file = st.camera_input("Take a picture")

if file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    detect_faces(opencv_image, "test/ex")
