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





#sys.path.append(str("/app/api"))
#sys.path.append(str("/app/utils"))
#print("this is the path src", str(os.listdir()))
#print("this is the path src", str(src_path))
#sys.path.append('/app/utils')

class StatefullMultiClassFBeta:

    """
    Custom Keras Fbeta metric used to calculate F beta during modeling
    """

    def __init__(self,
                name="state_full_binary_fbeta",
                beta=1,
                n_class=2,
                average="macro",
                epsilon=1e-7,
                **kwargs):

        # initializing an object of the super class
        super(StatefullMultiClassFBeta, self).__init__(name=name, **kwargs)

        # initializing state variables
        # initializing true positives
        self.tp = self.add_weight(name="tp", shape=(n_class,), initializer="zeros")

        # initializing actual positives
        self.actual_positives = self.add_weight(
            name="ap", shape=(n_class,), initializer="zeros"
        )
        # initializing predicted positives
        self.predicted_positives = self.add_weight(
            name="pp", shape=(n_class,), initializer="zeros"
        )

        # initializing other atrributes that won't be
        # changed for every object of this class
        self.beta_squared = beta**2
        self.n_class = n_class
        self.average = average
        self.epsilon = epsilon

    def update_state(self, ytrue, ypred, sample_weight=None):
        """
        Updates states during training
        """

        # casting ytrue and ypred as float dtype
        ytrue = tf.cast(ytrue, tf.float32)
        ypred = tf.cast(ypred, tf.float32)

        # finding the maximum probability in ypred
        max_prob = tf.reduce_max(ypred, axis=-1, keepdims=True)

        # making ypred one hot encoded so class with the maximum probability
        # will be encoded as 1 while others as 0
        ypred = tf.cast(tf.equal(ypred, max_prob), tf.float32)

        # updating true positives atrribute
        self.tp.assign_add(tf.reduce_sum(ytrue * ypred, axis=0))

        # updating predicted positives atrribute
        self.predicted_positives.assign_add(tf.reduce_sum(ypred, axis=0))

        # updating actual positives atrribute
        self.actual_positives.assign_add(tf.reduce_sum(ytrue, axis=0))

    def result(self):
        """
        Calculates and returns F beta score
        """

        self.precision = self.tp / (self.predicted_positives + self.epsilon)
        self.recall = self.tp / (self.actual_positives + self.epsilon)

        # calculating fbeta score
        self.fb = ((1 + self.beta_squared) * self.precision * self.recall / \
            (self.beta_squared * self.precision + self.recall + self.epsilon))

        if self.average == "weighted":
            return tf.reduce_sum(self.fb * self.actual_positives / \
                                tf.reduce_sum(self.actual_positives))

        elif self.average == "raw":
            return self.fb

        return tf.reduce_mean(self.fb)

    def reset_states(self):
        """
        Resets all states
        """

        self.tp.assign(tf.zeros(self.n_class))
        self.predicted_positives.assign(tf.zeros(self.n_class))
        self.actual_positives.assign(tf.zeros(self.n_class))


# custom metric used during training needed for prediction
dependencies = {
   'StatefullMultiClassFBeta': StatefullMultiClassFBeta
}

#params = load_params(os.path.join(src_path, 'params.yaml'))
#params = load_params( 'app/params.yaml')
#best_model = params.model_inference.best_model_name
#best_model_location = params.model_inference.best_model_location

model = tf.keras.models.load_model(os.path.join(src_path, "best_model.hdf5"), 
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


uploaded_file = st.file_uploader("Upload a picture of a person", type="jpg")
if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    #st.write(f"{prediction}")
    st.write("Classifying...")
    prediction = fake_detector(image)
    prediction_pct = prediction * 100

    label = np.argmax(prediction)
    formatted_pct = round(float(prediction_pct[0][label]),3)
    if label == 0:
        st.write(f"AI generated picture: model certainty of {formatted_pct}%")
    elif label ==1:
        st.write(f"Authentic picture: model certainty of {formatted_pct}%")

    st.pyplot(explainable_layers(image))

#only best model location until a place is found for images
#prediction = fake_detector(best_model_location)

#print(prediction)

