# import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from numpy.linalg import norm
from Feuture_extraction.feature_extraction import FeaturExtraction
feature_extraction = FeaturExtraction()
app = Flask(__name__)
feature_list = np.array(pickle.load(open('models/feature.pkl','rb')))
filenames = pickle.load(open('models/filenames.pkl','rb'))
neighbors = pickle.load(open('models/NearestNeighbour.pkl','rb'))
ResNet50 = pickle.load(open('models/resnet50.h5','rb'))


st.title('Reverse Image Search')
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())   # writing the bufferdata of image
        return 1
    except:
        return 0


uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = feature_extraction.feature_extractions(os.path.join("uploads",uploaded_file.name),ResNet50)
        # recommendention
        Distances, indices = neighbors.kneighbors(np.expand_dims(features, axis=0))

        
# make a columns to represent the output
        col1,col2,col3,col4,col5 = st.beta_columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occured in file upload")

if __name__=="__main__":
    app.run()