import streamlit as st
import keras
import glob
import cv2
import os
import numpy as np
import tensorflow as tf
import face_recognition
model=model = keras.models.load_model('femalevsmale_mobilenetv2_ft_80f')
st.set_page_config(page_title="PH", layout="wide")
st.markdown("<h1 style='text-align: center; color: grey;'>Gender Classification</h1>", unsafe_allow_html=True)
IMAGE_SIZE = [160, 160]
class_names = ["Female", "Male"]
def detect_faces(image, output_folder):

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use face_recognition library for face detection
    faces = face_recognition.face_locations(image_rgb)

    # Check if any faces are detected
    if len(faces) > 0:
        # Loop over the face locations
        for ( x, y, w, h) in faces:
            # Extract the face region
            face_img = image[x:w, h:y]

            # Construct the output file path
            output_path = os.path.join(output_folder, 'face.jpg')
            
            # Save the face ROI as a separate image
            cv2.imwrite(output_path, face_img)
            
            #get the label
            test_dataset = tf.keras.utils.image_dataset_from_directory("test",
                                                                    batch_size=1,
                                                                    image_size=IMAGE_SIZE,
                                                                    shuffle=False)
            # Retrieve a batch of images from the test set
            image_batch, label_batch = test_dataset.as_numpy_iterator().next()
            predictions = model.predict_on_batch(image_batch).flatten()

            # Apply a sigmoid since our model returns logits
            predictions = tf.nn.sigmoid(predictions)
            predictions = tf.where(predictions < 0.5, 0, 1)
            
            #remove the image
            os.remove(output_path)

            # Get the predicted gender label
            gender_label = class_names[predictions[0]]  # Implement a function to interpret the model's output
            
            # Determine the color for the bounding box based on the predicted gender
            if gender_label == 'Female':
                color = (255, 0, 255)  # Pink color
            else:
                if gender_label == 'Male':
                    color = (255, 0, 0)  # Blue color

            # Draw bounding box on the image
            thickness = int((image.shape[0] + image.shape[1]) / 600)  # Calculate thickness based on image size
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

        # Convert the image from BGR to RGB for Matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        col1, col2, col3 = st.columns(3)
        with col2:
            st.image(image_rgb, caption="Face(s) detected")

        with col1:
            # Set the square size
            square_size = 50

            # Create a blue square image
            square_image = np.zeros((square_size, square_size, 3), dtype=np.uint8)
            square_image[:, :] = [0, 0, 255]  # Blue color

            # Display the square image and text
            st.image(square_image, caption='Male', width=square_size)
            # Create a pink square image
            square_image = np.zeros((square_size, square_size, 3), dtype=np.uint8)
            square_image[:, :] = [255, 0, 255]  # pink color

            # Display the square image and text
            st.image(square_image, caption='Female', width=square_size)

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
