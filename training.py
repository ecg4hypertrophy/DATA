import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import os
import json
from collections import Counter
from datetime import date
from tqdm import tqdm_notebook
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix, roc_auc_score
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Conv1D, MaxPooling1D, GlobalAveragePooling1D, SeparableConv1D, SeparableConv2D, Reshape, Multiply, GlobalAveragePooling2D, Dropout, add, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from ecg_preprocessing import normalize, wavelet_noising, standardization

# Model building function
def model_build(params):
    def SE_X_Res_Module(input_data, n_filters, model_name, filter_size, index):     
        x2 = SeparableConv1D(filters=n_filters, kernel_size=filter_size, padding='same', name=f'{model_name}_{index}_SeparableConv1D_1')(input_data)        
        x2 = BatchNormalization(name=f'{model_name}_{index}_BN_1')(x2)  
        x2 = Activation('relu', name=f'{model_name}_{index}_relu_1')(x2)      
        x2 = SeparableConv1D(filters=n_filters, kernel_size=filter_size, padding='same', name=f'{model_name}_{index}_SeparableConv1D_3')(x2)        
        x2 = MaxPooling1D(name=f'{model_name}_{index}_max_pool_1', padding='same')(x2) 

        x3 = SeparableConv1D(filters=n_filters, kernel_size=filter_size, padding='same', name=f'{model_name}_{index}_SeparableConv1D_4')(input_data)
        x3 = MaxPooling1D(name=f'{model_name}_{index}_max_pool_2', padding='same')(x3) 
        
        add_s = add([x2, x3], name=f'{model_name}_{index}_add')
        
        # Squeeze-and-Excitation (SE) Block
        x4 = GlobalAveragePooling1D(name=f'{model_name}_{index}_GlobalAveragePooling1D')(add_s)
        x4 = Dense(n_filters//1, name=f'{model_name}_{index}_FC1')(x4)
        x4 = Activation('relu', name=f'{model_name}_{index}_relu_3')(x4)
        x4 = Dense(n_filters, name=f'{model_name}_{index}_FC2')(x4)
        x4 = Activation('sigmoid', name=f'{model_name}_{index}_relu_4')(x4)
        x4 = Reshape((1, n_filters), name=f'{model_name}_{index}_reshape')(x4)
        mul = Multiply()([add_s, x4])

        return add_s

    input_ecg = Input(shape=(5000, 12), name='input_ecg') 
    input_cv = Input(shape=(128, 128, 12), name='input_cv') 
    
    x_o = Conv1D(filters=32, kernel_size=15, strides=1, padding='same', kernel_initializer=params["conv_init"], name='conv_1d_1')(input_ecg)
    x_o = BatchNormalization(name='BN_1d_1')(x_o)
    x_o = Activation('relu', name='relu_1d_1')(x_o)
    x_o = MaxPooling1D(pool_size=2)(x_o)

    for i in range(6):
        x_o = SE_X_Res_Module(x_o, n_filters=params["conv_num_filters"][i], model_name="SE_X_Res_Module", filter_size=15, index=i + 1) 

    x_o = GlobalAveragePooling1D(name='average_pooling_1d')(x_o)
    
    x_c = Conv1D(filters=32, kernel_size=15, strides=1, padding='same', kernel_initializer=params["conv_init"], name='conv_2d_1')(input_cv)
    x_c = BatchNormalization(name='BN_2d_1')(x_c)
    x_c = Activation('relu', name='relu_2d_1')(x_c)
    x_c = MaxPooling1D(pool_size=2)(x_c)

    for i in range(6):
        x_c = SE_X_Res_Module(x_c, n_filters=params["conv_num_filters"][i], model_name="SE_X_Res_Module_2", filter_size=15, index=i + 1)  

    x_c = GlobalAveragePooling2D(name='average_pooling_2d')(x_c)
    
    x = concatenate([x_o, x_c], axis=-1)  
    x = Dense(params["dense_neurons"], name='FC1')(x)
    x = BatchNormalization(name='BN_7')(x)
    x = Activation('relu', name='relu_7')(x) 

    x = Dropout(rate=params["dropout"])(x) 
    x = Dense(5, activation='sigmoid', name='output')(x)

    model = Model(inputs=[input_ecg, input_cv], outputs=x)
    return model

# Custom metrics
def acc(y_true, y_pred):
    epsilon = 1.e-7
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.round(tf.clip_by_value(y_pred, epsilon, 1. - epsilon))
    y_judges = tf.reduce_all(tf.equal(y_true, y_pred), 1)
    y_onehot = tf.where(y_judges, tf.ones_like(y_judges), tf.zeros_like(y_judges))
    true_num = tf.count_nonzero(y_onehot)
    all_num = tf.count_nonzero(tf.ones_like(y_onehot))
    acc = true_num / all_num
    return acc
    
def f1(y_true, y_pred):
    epsilon = 1.e-7
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.round(tf.clip_by_value(y_pred, epsilon, 1. - epsilon))

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# Data generator
class DataGenerator(Sequence):
    def __init__(self, data_ids, labels, batch_size, n_classes, shuffle=True):
        self.data_ids = data_ids
        self.labels = labels
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.data_ids) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        data_ids_temp = [self.data_ids[k] for k in indexes]
        X, y = self.__data_generation(data_ids_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_ids_temp):
        ecg = np.empty((self.batch_size, 5000, 12))
        cvecg = np.empty((self.batch_size, 128, 128, 12))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)
        
        for i, ID in enumerate(data_ids_temp):        
            filename = os.path.basename(ID)
            basename, suffix = os.path.splitext(filename)
            cvname = basename + '.png'

            ecg[i] = standardization(normalize(np.load(os.path.join('/mnt/data1/ECG_data/denoise_data/', filename))))
            
            for j in range(12):
                cvecg[i][:, :, j] = cv.resize(cv.imread(os.path.join(f'/mnt/data1/ECG_data/ECG-cv/ECG-cv{j}', cvname), cv.IMREAD_GRAYSCALE), (128, 128))
            
            y[i] = self.labels[ID]
        return [ecg, cvecg], y

# Model training function
def model_train(model, train_id, train_label, val_id, val_label, params):
    adam = Adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', metrics=[f1, acc], optimizer=adam)  
    
    my_callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, verbose=2),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.0000001, verbose=1)    ]
    
    model.fit_generator(
        generator=DataGenerator(train_id, train_label, batch_size=64, n_classes=params["disease_num"]),
        epochs=50,
        validation_data=DataGenerator(val_id, val_label, batch_size=64, n_classes=params["disease_num"]),
        steps_per_epoch=int(len(train_label) / 64),
        callbacks=my_callbacks,
        class_weight='auto'
    )

# Load parameters and train the model
params = json.load(open('config.json', 'r'))

model = model_build(params)
model_train(model, train_id, train_label, val_id, val_label, params)


# Make predictions
test_pos_predict_1 = model.predict(X_test)
test_predict_onehot_1 = (test_pos_predict_1 >= 0.5).astype(int)

# Adjust thresholds for specific classes
test_predict_onehot_1[:, 0] = (test_pos_predict_1[:, 0] >= 0.578).astype(int)
test_predict_onehot_1[:, 1] = (test_pos_predict_1[:, 1] >= 0.449).astype(int)
test_predict_onehot_1[:, 2] = (test_pos_predict_1[:, 2] >= 0.482).astype(int)
test_predict_onehot_1[:, 3] = (test_pos_predict_1[:, 3] >= 0.308).astype(int)
test_predict_onehot_1[:, 4] = (test_pos_predict_1[:, 4] >= 0.273).astype(int)

# Evaluate model performance
print('Absolute Accuracy:', accuracy_score(y_test, test_predict_onehot_1))
