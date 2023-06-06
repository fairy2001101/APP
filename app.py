import streamlit as st
import keras
model = keras.models.load_model('femalevsmale_mobilenetv2_ft_80f')
import cv2
import os
import numpy as np


st.set_page_config(page_title="PH",page_icon=":tada:")
file = st.file_uploader("Please choose an Image")
# file = st.camera_input("Take a picture")

if file is not None:
     # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    # Load the pre-trained cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Load the registered photo
    registered_photo = opencv_image
    #cv2.imread(r"C:\Users\ezzin\Desktop\Python\faces detection\IMG_20211231_174903.jpg")

    # Convert the registered photo to grayscale for face detection
    gray_registered = cv2.cvtColor(registered_photo, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale registered photo
    faces = face_cascade.detectMultiScale(gray_registered, scaleFactor=1.2, minNeighbors=5, minSize=(200,200))

    # Specify the output directory to save the detected faces
    output_directory = "test/ex"

    # Save the detected faces as separate images
    for i, (x, y, w, h) in enumerate(faces):
        face_roi = registered_photo[y:y+h, x:x+w]  # Extract the region of interest (ROI) of the face
        # Construct the output file path
        output_path = os.path.join(output_directory, f'face_{i}.jpg')
        
        cv2.imwrite(output_path, face_roi)  # Save the face ROI as a separate image

        # Draw rectangles around the detected faces
        cv2.rectangle(registered_photo, (x, y), (x + w, y + h), (0, 255, 0), 2)    

    # Convert the BGR image to RGB for proper display with matplotlib
    registered_photo_rgb = cv2.cvtColor(registered_photo, cv2.COLOR_BGR2RGB)

    import tensorflow as tf
    import numpy as np
    import os
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

    import glob

    image_files = glob.glob("test/ex/*")
    num_images = len(image_files)


    for i in range(num_images):
        st.image(image_batch[i].astype("uint8"),caption=class_names[predictions[i]])
    for image in image_files:
        os.remove(image)
