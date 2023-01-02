import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors



class BuilModel:

    def Resnet(self):
        model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
        model.trainable = False
        # passed the model into sequential model
        model = tensorflow.keras.Sequential([
            model,
            GlobalMaxPooling2D()
        ])
        return model
    


    def neighbour(self,features,feature_list):
        neighbo = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbo.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices