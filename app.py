import streamlit as st
import keras
import glob
model = keras.models.load_model('femalevsmale_mobilenetv2_ft_80f')
import cv2
import os
import numpy as np
import tensorflow as tf
st.set_page_config(page_title="PH",layout="wide")
st.markdown("<h1 style='text-align: center; color: grey;'>Gender Classification</h1>", unsafe_allow_html=True)
def detect_faces(image, output_folder):
    # Load pre-trained model files
    prototxt_path = "deploy.prototxt.txt"
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"

    # Load the model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Set the blob as input to the network
    net.setInput(blob)

    # Forward pass through the network to detect faces
    detections = net.forward()

    # Check if any faces are detected
    confidence_threshold = 0.5
    num_detections = detections.shape[2]
    if any(detections[0, 0, i, 2] > confidence_threshold for i in range(num_detections)):
        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections by confidence threshold
            if confidence > 0.5:
                # Get the coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Extract the face ROI (Region of Interest) without the green square
                face_roi = image[startY:endY, startX:endX]

                # Generate a unique filename for saving the face image
                filename = f"face_{i}.jpg"
                output_path = os.path.join(output_folder, filename)

                # Save the face image without the green square
                cv2.imwrite(output_path, face_roi)

                # Draw a green square around the face on the original image
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Convert BGR image to RGB for displaying with plt
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        col1, col2, col3 = st.columns(3)
        with col2:
            st.image(image_rgb,caption="Face(s) detecter")
        IMAGE_SIZE=[160,160]
        class_names=["Female","Male"]
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
                st.image(image_batch[i].astype("uint8"),caption=class_names[predictions[i]])
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
