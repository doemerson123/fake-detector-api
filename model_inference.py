import os
import sys
import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM
tf.config.run_functions_eagerly(True)
import numpy as np     
import matplotlib.pyplot as plt
from pathlib import Path
#import cv2
import streamlit as st 
src_path = Path(__file__).parent.parent.parent.resolve()
#print(str(os.listdir())
src_path.joinpath(src_path, "api")




#sys.path.append(str("/app/api"))
#sys.path.append(str("/app/utils"))
#print("this is the path src", str(os.listdir()))
#print("this is the path src", str(src_path))
#sys.path.append('/app/utils')

from app.utils.custom_metrics import StatefullMultiClassFBeta
from app.utils.load_params import load_params

# custom metric used during training needed for prediction
dependencies = {
   'StatefullMultiClassFBeta': StatefullMultiClassFBeta
}

#params = load_params(os.path.join(src_path, 'params.yaml'))
params = load_params( 'app/params.yaml')
best_model = params.model_inference.best_model_name
best_model_location = params.model_inference.best_model_location

model = tf.keras.models.load_model(os.path.join(best_model_location, best_model), 
                                custom_objects=dependencies)

def preprocess_image(image):
    '''
    Matches preprocessing steps in model training for inference
    image_path: path to image
    returns: tensor with shape (1, 200, 200, 3)
    '''
#    image = tf.io.read_file(image_path)
    image = tf.convert_to_tensor(image)
    #image = tf.keras.utils.img_to_array(image)
    #image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (200,200))
    image = tf.expand_dims(image, axis=0)
    return image

@tf.function
def fake_detector(inference_image):
    '''
    Predicts if image is real or fake
    image_path: path to image
    returns: prediction 0 = Fake, 1 = Real
    '''
    
    #image = preprocess_image(os.path.join("app/api/Fake 4.jpg"))
    inference_image = preprocess_image(inference_image)
    return model.predict(inference_image)

@tf.function
def explainable_layers(explainable_image):
    explainable_image = preprocess_image(explainable_image)
    explainable_image = tf.reshape(explainable_image, [200,200,3])
    explainable_image = np.array(explainable_image)
    
    
    conv_layer_list = []

    # list all convolution layers
    for layer in model.layers:
        if 'conv2d' in str(layer.name): 
            conv_layer_list.append(layer.name)

    explainer = GradCAM()
    
    
    layer_label_counter = len(conv_layer_list)-1
 
    fig, axs = plt.subplots(len(conv_layer_list),1, figsize=(4, 4), dpi=600, facecolor='w', edgecolor='k')
    axs = axs.ravel()

    for conv_layer, conv_layer_depth in zip(conv_layer_list, range(len(conv_layer_list))):
        grid1 = explainer.explain(([explainable_image], None), model, class_index=1, layer_name =conv_layer, image_weight=.4 )
        axs[conv_layer_depth].imshow(explainable_image / 255.)
        axs[conv_layer_depth].imshow(grid1, alpha=0.9, cmap='bwr')
        axs[conv_layer_depth].set_title(f'Convolution Layer: {conv_layer, conv_layer_depth}', size=9)
        axs[conv_layer_depth].set_xticks([])
        axs[conv_layer_depth].set_yticks([])
    plt.tight_layout(pad=0.3)
    fig.set_figheight(12)
    fig.set_figwidth(12)
    plt.subplots_adjust(wspace=0, hspace=.1)
    return(fig)

#only best model location until a place is found for images
#prediction = fake_detector(best_model_location)

#print(prediction)