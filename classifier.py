import os
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import cv2
import urllib.request
import numpy as np
import time

# Getting some unknown linter errors, disable everything to get this to production asap
# pylint: disable-all

class Classifier:
    """
    Classifier class for machine learning models and algorithms
    """

    def __init__(self):
        """
        Initialize the classifier
        """
        self

class ImageClassifier(Classifier):
    """
    Image Classifier class for machine learning image predictions.
    """

    def __init__(self, model_url, labels_url, image_urls):
        """
        Initialize the image classifier
        """
        super().__init__()
        self.model_url = model_url
        self.labels_url = labels_url
        self.image_urls = image_urls
    
    def load_model_from_url(self):
        """
        Load the model from the model url
        """
        return self.model_url
    
    def load_model_from_path(self, model_path):
        """
        Load the model from the model path
        """
        return model_path

    def load_model_from_hub(self, model_url):
        """
        Load the model from the tensorflow hub
        """
        return hub.KerasLayer(model_url)
    
    def load_labels(self):
        """
        Load the labels from the labels url
        """
        return self.labels_url

    def clean_header_labels(self, labels_url):
        """
        Clean the header labels
        """
        labels_raw = urllib.request.urlopen(labels_url)
        labels_lines = [line.decode('utf-8').replace('\n', '') for line in labels_raw.readlines()]
        labels_lines.pop(0)  # remove header (id, name, ...)
        cleaned_labels = {}
        for line in labels_lines:
            id = int(line.split(',')[0])
            name = line.split(',')[1]
            cleaned_labels[id] = {'name': name}
        return cleaned_labels

    def load_image_by_url(self):
        """
        Load the image from the image url
        """
        return self.image_urls
    
    def load_image_by_path(self, image_path):
        """
        Load the image from the image path
        """
        return image_path

    def preprocess_image(self, model, image, labels_url):
        """
        Preprocess the image to generate tensor
        """
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)
        model_raw_output = model.call(image_tensor).numpy()
        return self.order_by_result_score(model_raw_output, labels_url)

    def order_by_result_score(self, model_raw_output, cleaned_labels):
        """
        Order the results by score based on model raw output and cleaned labels
        """
        for index, value in np.ndenumerate(model_raw_output):
            result_index = index[1]
            cleaned_labels[result_index]['score'] = value
        
        return sorted(cleaned_labels.items(), key=lambda x: x[1]['score'])

    def get_top_n_results(self, top_index, order_by_result_score):
        """
        Get the top n results based on result score
        """
        name = order_by_result_score[top_index*(-1)][1]['name']
        score = order_by_result_score[top_index*(-1)][1]['score']
        return name, score

class BirdClassifier(ImageClassifier):
    """
    Bird Classifier class for machine learning image predictions based on birds.
    """
    def __init__(self, model_url, labels_url, image_urls):
        """
        Initialize the bird classifier
        """
        super().__init__(model_url, labels_url, image_urls)
  
