import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
import shap
import matplotlib.pyplot as plt

# Define the custom F1 score metric
def f1(y_true, y_pred):
    epsilon = 1.e-7
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.round(tf.clip_by_value(y_pred, epsilon, 1. - epsilon))
    
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())

    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    return 2 * ((precision_value * recall_value) / (precision_value + recall_value + K.epsilon()))

# Load the model with the custom F1 metric
model = load_model('./binary_model.h5', custom_objects={'f1': f1})
model.summary()

# Function to map data to model inputs
def map2layer(ecg, cvecg):
    feed_dict = dict(zip(model.inputs, [ecg, cvecg]))
    return K.get_session().run(model.inputs, feed_dict)

# Initialize SHAP GradientExplainer
tf.compat.v1.disable_eager_execution()
e = shap.GradientExplainer(model, map2layer(ecg, cvecg), local_smoothing=0)

# Compute SHAP values
svs_ecg, svs_cvecg, y_scores = [], [], []
for i in range(len(ecg)):
    sv_ecg, sv_cvecg = [], []
    test_pos_predict = model.predict([ecg[i:i+1], cvecg[i:i+1]])
    y_scores.append(test_pos_predict)
    
    sv = e.shap_values([np.expand_dims(ecg[i], axis=0), np.expand_dims(cvecg[i], axis=0)])
    for sv_val in sv:
        sv_ecg.append(sv_val[0])
        sv_cvecg.append(sv_val[1])
        
    svs_ecg.append(np.array(sv_ecg))
    svs_cvecg.append(np.array(sv_cvecg))


# Plotting SHAP values
leads = np.array(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
cmap = plt.cm.Blues
ys = []

for j in range(12):  # For each lead
    sum_ = 0
    for i, ecg_data in enumerate(ecg):
        if y_test[i] == 1.0 and test_predict_onehot[i] == [1]:  # Filter correctly predicted samples
            label_idx = np.argmax(y_scores[i])
            sv_data = np.squeeze(svs_ecg[i][label_idx])
            sv_data = np.swapaxes(sv_data, 0, 1)
            ecg_data = np.swapaxes(ecg_data, 0, 1)
            sv_data = sv_data[:, -5000:]
            
            shap_value = sv_data[j]
            shap_mean = shap_value.mean()
            shap_std = shap_value.std()
            
            for k in range(5000):
                if abs((sv_data[j][k] - shap_mean) / shap_std) >= 1:
                    sum_ += abs(sv_data[j][k])
    
    ys.append(sum_)

ys = np.array(ys) / np.sum(ys)
ys = np.array([ys])  # Reshape for plotting

# Plotting
fig, axs = plt.subplots(figsize=(7, 5))
im = axs.imshow(ys, cmap=cmap)
axs.figure.colorbar(im, ax=axs)

xlabels = leads
ylabels = ['RVH']
axs.set_xticks(np.arange(len(xlabels)))
axs.set_yticks(np.arange(len(ylabels)))
axs.set_xticklabels(xlabels)
axs.set_yticklabels(ylabels)

thresh = ys.max() / 2
for i in range(ys.shape[0]):
    for j in range(ys.shape[1]):
        axs.text(j, i, format(ys[i, j], '.3f'),
                 ha='center', va='center',
                 color='white' if ys[i, j] > thresh else 'black')

fig.tight_layout()
plt.show()
plt.clf()
