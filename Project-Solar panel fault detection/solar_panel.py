import streamlit as st
import tensorflow as tf
from keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from keras.preprocessing import image
import numpy as np
import cv2
from IPython.display import Image, display
from PIL import Image, ImageOps
import matplotlib.pyplot as plt 
import tensorflow as hub
from tensorflow.keras import preprocessing
from keras.models import load_model
from tensorflow.keras.activations import softmax
import h5py
import os
from tempfile import NamedTemporaryFile
from django.core.files.storage import default_storage

page_bg_img='''
<style>
.stApp{
    background-image: url("https://img.resized.co/businesspost/eyJkYXRhIjoie1widXJsXCI6XCJodHRwOlxcXC9cXFwvaW1lbmdpbmUucHVibGljLnByb2Quc2JwLmluZm9tYWtlci5pbz91dWlkPTg4ZjFmNTFlLTc4YTItNTQxOC1iM2U3LTA4MTJkOTliMDhkYiZ0eXBlPXByZXZpZXcmcT04MCZoZWlnaHQ9MjE3MC43ODY1MTY4NTM5JndpZHRoPTM4NjQmZnVuY3Rpb249Y3JvcHJlc2l6ZSZ4PTAmeT0wLjE1NTQzMDcxMTYxMDQ4Njg4JmNyb3Bfdz0wLjk5OTk5OTk5JmNyb3BfaD0wLjg0MjY5NjYxOTIxMzQ4XCIsXCJ3aWR0aFwiOjEyMDAsXCJoZWlnaHRcIjo2MjcsXCJkZWZhdWx0XCI6XCJodHRwOlxcXC9cXFwvczMtZXUtd2VzdC0xLmFtYXpvbmF3cy5jb21cXFwvc3RvcmFnZS5wdWJsaXNoZXJwbHVzLmllXFxcL21lZGlhLmJ1c2luZXNzcG9zdC5pZVxcXC9zYnAtbm8taW1hZ2UucG5nXCIsXCJvcHRpb25zXCI6W119IiwiaGFzaCI6Ijg1ZGUyMjVlYjI4NGM4NTRjZWEzOWE1Y2Y1OWZlZmJlMThmYWY0OGYifQ==/excess-electricity-from-solar-panels-not-being-used-in-the-way-it-should-be.84269661921348");
    background-size: cover;
}
</style>
'''
header = st.container()
with header:
    st.title('Solar Panel Fault Detection!') 
    st.text('In this project we look Solar panel is faulty or Not.')
st.markdown(page_bg_img, unsafe_allow_html = True)
filename = (r'C:\Users\Solan\OneDrive\Desktop\Rohit Ghadage\Project Dataa science\Solar panel fault detection\dataset1\test_set\testing')
# st.write('You selected `%s`' % filename)
imagess = st.file_uploader("Choose a jpeg file")
if imagess is filename:
    path_in = filename + imagess.name
    print(path_in)
# st.write(filename + "/" + imagess.name)
path_in = filename + "/"+imagess.name
# else:
#     path_in = None
# s = st.image(temp_file)
#a = st.write(imagess.read())

model = tf.keras.models.load_model('SolarPanelFaultDetection.h5')
# ss = a or b + "/" + path_in
image_path = path_in
test_image = image.load_img(image_path, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print (result)

#training_set.class_indices
if result[0][0] == 1:
    prediction = 'NoDefect'
    print('Nodefect')
    image = cv2.imread(image_path)
    cv2.putText(image, "No Defect predicted", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    #cv2.waitKey(300)
    cv2.imshow("Predictions", image)
    st.image(image, caption = prediction )
    #cv2.waitKey(30000)
else:
    prediction = 'Defect'
    print('Defect')

    image = cv2.imread(image_path)
    cv2.putText(image, "Defect predicted", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #cv2.waitKey(300)
    cv2.imshow("Predictions", image)
    st.image(image, caption = prediction )
    #cv2.waitKey(30000)

