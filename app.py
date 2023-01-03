from flask import Flask,url_for,request,render_template,jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
# import tensorflow
# from numpy.linalg import norm
from Feuture_extraction.feature_extraction import FeaturExtraction
from tensorflow.keras.models import  load_model
feature_extraction = FeaturExtraction()
app = Flask(__name__)
# feature_list = np.array(pickle.load(open('models/feature.pkl','rb')))
filenames = pickle.load(open('models/files.pkl','rb'))
neighbors = pickle.load(open('models/NearestNeighbour.pkl','rb'))
ResNet50 = load_model('models/resnet50.h5')

# to get the file path of all recommend  images
def getfilepath(indices):
    file = []
    for i in  indices[0][1:]:
        file.append(filenames[i])
    return file


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/result' , methods=['post'])
def result():
    if request.method == 'POST':   
       uploaaded_file = request.files['file']

       # to save the file
       basepath = os.path.dirname(__file__)  # to get the bas path
       file_path = os.path.join(basepath, 'uploads', secure_filename(uploaaded_file.filename))   # complete path
       uploaaded_file.save(file_path)

       # to extract the features from image
       features = feature_extraction.feature_extractions(file_path,ResNet50)
       # ### to get the neghbours

       Distances, indices = neighbors.kneighbors(np.expand_dims(features, axis=0))
       file_paths = getfilepath(indices)

                
    return render_template('result.html',file1 = file_paths[0],
        file2 = file_paths[1],
        file3 = file_paths[2],
        file4 = file_paths[3],
        file5 = file_paths[4],
        loaded_file = file_path)

if __name__  == "__main__":
    app.run(debug=True)