#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:18:30 2018

@author: Manish Bhanga
"""


from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.layers import  Activation, Input, Lambda
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
import keras.backend as K
import pandas as pd
import speechpy as sp
import scipy.io.wavfile as wav
from collections import Counter
from itertools import compress
from flask import Flask, request, jsonify
import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='voice_authentication%s.log'%(str(datetime.now())),
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['wav'])
REG_USER_DIR = '/root/Voice_Authentication/Users/'
TEST_USER_DIR = '/root/Voice_Authentication/Test/'

reg_features = []
reg_labels = []

def allowed_file(filename):
    return filename[-3:].lower() in ALLOWED_EXTENSIONS

def create_base_network(input_dim):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), padding='same',
                     input_shape = input_dim))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    #model.add(Dense(10))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(128))
    #model.add(Activation('softmax'))
    
    return model

def euclidean_distance(vects):
    x, y = vects

    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
  
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))



def compute_accuracy(predictions, labels):
    
    return labels[predictions.ravel() < 0.5].mean()


input_dim = (199,40,3)

base_network = create_base_network(input_dim)
base_network.load_weights('my_model_weights_siamese_3.h5')
input_a = Input(shape=input_dim)
input_b = Input(shape=input_dim)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)

model.load_weights('my_model_weights_siamese_2.h5')

@app.route('/register', methods=['POST'])
def Register_User():
    if request.method == 'POST':
        logging.info("Registering user")
    
        global reg_features
        global reg_labels

        
        
        file = request.files['wav_file']
        if file and allowed_file(file.filename):
            file_ = file.filename
            name = file_.split('.')[0]
            if os.path.exists(REG_USER_DIR+name):
                name = name+'1'
                
            try:
                
                subprocess.call("mkdir {a}{b};".format(a = REG_USER_DIR, b = name), shell = True)    
                app.config['UPLOAD_FOLDER'] = REG_USER_DIR+name
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_))   
            
                subprocess.call("sox  {a}{b}/{d} {a}{b}/out%03d.wav trim 0 2 : newfile : restart;"
                                "rm {a}{b}/{d};".format(a = REG_USER_DIR, b = name, d = file_), shell = True)
                
            except Exception as e:
                logging.info("Error in creating directory")
                logging.error(e)
                
        else:
            logging.info("File not found or file not wav")
            return jsonify("Service only accepts single channel wav file")
    
        wav_folder = REG_USER_DIR+name
    
        for wav_file in os.listdir(wav_folder):
            try:
                filename = wav_file.split('.')[0]
                fs , signal = wav.read(wav_folder+'/'+wav_file)
                if signal.shape[1] == 2:
                    x = pd.DataFrame(signal)
                    signal = np.array((x[0] + x[1]) / 2)
                    signal = np.round(signal)
                    
                feature = sp.feature.lmfe(signal=signal, sampling_frequency=fs,frame_length=0.02,frame_stride=0.01,num_filters=40)
                plt.imsave(wav_folder+'/'+filename+'.png',feature)
            except Exception as e:
                logging.info("Error in png savings")
                logging.error(e)
            
            

        spectograms = []
        spect_read = []
        spectograms_ids = []
        for folder in os.listdir(REG_USER_DIR):
            print(folder)
            spect_dir = REG_USER_DIR+folder+'/'
            for file_ in os.listdir(spect_dir):
                if file_.endswith('png'): 
                    try:
                        x = plt.imread(spect_dir+file_)
                        x = np.array(x)[:,:,:3]
                    except Exception as e:
                        logging.info("Error in reading png")
                        logging.error(e)
                        continue
                    if str(x.shape) == '(199, 40, 3)': 
                        spect_read.append(x)
                        spectograms_ids.append(folder)
                        spectograms.append(file_)
    
        reg_features = spect_read
        reg_labels = spectograms_ids
    
    else:
        logging.info("Request was not POST")
        return jsonify("Request not POST")
        
    logging.info("User Registered")
    return jsonify("User Registered")
    
@app.route('/test', methods=['POST'])
def Test_User():
    if request.method == 'POST':
        logging.info("Testing user")
        global reg_features
        global reg_labels
        
        if reg_features == []:
            return jsonify("No user registered")
        
        file = request.files['wav_file']
        #print(file)
        if file and allowed_file(file.filename):
            #print '**found file', file.filename
            file_ = file.filename
            name = file_.split('.')[0]
            if os.path.exists(TEST_USER_DIR+name):
                name = name+'1'
                
            try:
                
                subprocess.call("mkdir {a}{b};".format(a = TEST_USER_DIR, b = name), shell = True)    
                app.config['UPLOAD_FOLDER'] = TEST_USER_DIR+name
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_))   
            
                subprocess.call("sox  {a}{b}/{d} {a}{b}/out%03d.wav trim 0 2 : newfile : restart;"
                                "rm {a}{b}/{d};".format(a = TEST_USER_DIR, b = name, d = file_), shell = True)
                
            except Exception as e:
                logging.info("Error in creating directory")
                logging.error(e)
                
        else:
            logging.info("File not found or file not wav")
            return jsonify({"Service only accepts single channel wav file"})


        wav_folder = TEST_USER_DIR+name
    
        for wav_file in os.listdir(wav_folder):
            try:
                filename = wav_file.split('.')[0]
                fs , signal = wav.read(wav_folder+'/'+wav_file)
                if signal.shape[1] == 2:
                    x = pd.DataFrame(signal)
                    signal = np.array((x[0] + x[1]) / 2)
                    signal = np.round(signal)
                    
                feature = sp.feature.lmfe(signal=signal, sampling_frequency=fs,frame_length=0.02,frame_stride=0.01,num_filters=40)
                plt.imsave(wav_folder+'/'+filename+'.png',feature)
            except Exception as e:
                logging.info("Error in png saving")
                logging.error(e)
        
        predictions = []
        for test_png in os.listdir(wav_folder):
            if test_png.endswith('png'): 
                try:
                    feat = plt.imread(wav_folder+'/'+test_png)
                    feat = np.array(feat)[:,:,:3]
                except Exception as e:
                    logging.info("Error in reading png")
                    logging.error(e)
                    continue
                if str(feat.shape) == '(199, 40, 3)': 
                    pred = []
                    for reg_feat in reg_features:
                        pred.append(model.predict([np.expand_dims(feat,0), np.expand_dims(reg_feat,0)])[0][0])

                    predictions.append(Counter((list(compress(reg_labels,[np.array(pred).ravel() < 0.4][0])))).most_common(1)[0][0])
                else:
                    continue
    
        target_data = Counter(predictions)
        target = target_data.most_common(1)[0][0]        
    else:
        logging.info("Request not POST")
        return jsonify({"Request not POST"})
    
    return jsonify(target)



if __name__ == '__main__':
	app.run(debug=False,use_reloader = False, host = '0.0.0.0')