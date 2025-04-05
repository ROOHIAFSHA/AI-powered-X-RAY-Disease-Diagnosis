

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------


import gradio as gr
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os
import sys


# Add the parent folder to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.config import *

# Load the trained CNN model using the path from config
cnn_model = load_model(MODEL_LOC)


def make_prediction(test_image):
    test_image = image.load_img(test_image, target_size=(224, 224))  # Directly load using file path
    test_image = image.img_to_array(test_image) / 255.
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn_model.predict(test_image)
    return {"Normal": str(result[0][0]), "Pneumonia": str(result[0][1])}


image_input = gr.Image(type="filepath")

title = "AI Powered X-Ray Disease Diagnosis"
description = "This application uses a Convolutional Neural Network (CNN) model to predict whether " \
              "a chosen X-ray shows if the person has pneumonia disease or not. To check the model " \
              "prediction, here are the true labels of the provided examples below: the first 4 images " \
              "belong to normal whereas the last 4 images are of pneumonia category. More specifically, " \
              "the 5th and 6th images are of type viral pneumonia infection in nature whereas he last 2 images " \
              "are of bacterial infection in nature."

gr.Interface(fn=make_prediction,
             inputs=image_input,
             outputs="label",
             examples=[["image1_normal.jpeg"],
                       ["image2_normal.jpeg"],
                       ["image3_normal.jpeg"],
                       ["image4_normal.jpeg"],
                       ["image1_pneumonia_virus.jpeg"],
                       ["image2_pneumonia_virus.jpeg"],
                       ["image1_pneumonia_bacteria.jpeg"],
                       ["image2_pneumonia_bacteria.jpeg"]],
             title=title,
             description=description) \
    .launch()
