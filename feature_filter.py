from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras import optimizers
from keras.applications.vgg19 import preprocess_input
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
import cv2
import os

from common.utils import ImageSlice, ImageSliceInMemory

class FeatureExtrctor(object):
    """docstring for FeatureExtrctor"""
    def __init__(self, gpu_num = 1, gpus = "0"):
        self.gpu_num = gpu_num
        self.gpus = gpus

        self.model = None

    def build_model(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
        with tf.device('/cpu:0'):

        # build base model body
            input_tensor = Input((224, 224, 3))
            x = Lambda(vgg19.preprocess_input)(input_tensor)
        
            base_model = VGG19(input_tensor=x, weights=None, include_top=False)
        
        # build temp model for weights assign
            m_out = base_model.output
            flatten = Flatten(name='flatten')(m_out)
            fc1 = Dense(4096, activation='relu', name='fc1')(flatten)
            drop_fc1 = Dropout(1.0, name='drop_fc1')(fc1)
            fc2 = Dense(512, activation='relu', name='fc2')(drop_fc1)
            drop_fc2 = Dropout(1.0, name='drop_fc2')(fc2)
            predictions = Dense(2, activation='softmax', name='predictions')(drop_fc2)
            
            model_weights = Model(inputs=base_model.input, outputs=predictions)
            model_weights.load_weights("vgg19_finetune.h5")
        
        # build real model for feature extraction
        	input_tensor = Input((512))
        	drop_fc2 = Dropout(1.0, name='drop_fc2')(input_tensor)
            predictions = Dense(2, activation='softmax', name='predictions')(drop_fc2)
            
            
            model = Model(inputs=input_tensor, outputs=predictions)


            pretrained_weights = model_weights.get_layer(index=29).get_weights()
            print(model_weights.get_layer(index=29).name)
            model.get_layer(index=1).set_weights(pretrained_weights)
            model.summary()
        
        parallel_model = multi_gpu_model(model, gpus = self.gpu_num)
        #parallel_model.summary()

        print("    build completed!")

        self.model = parallel_model

    def feature_filter(self, features):
    	features = features.reshape(128*128, 512)
        result_all = self.model.predict(features, batch_size=128, verbose=1)

        for i, feature in enumerate(features):
        	if (1 == np.argmax(result_all[i])):
        		features[i] = np.zeros((512))

        features = features.reshape(128,128, 512)

        return features

        

