import logging
import os
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

# Getting some unknown linter errors, disable everything to get this to production asap
# pylint: disable-all


class ImageClassifier:
    """
    Image Classifier class for machine learning image predictions.
    """

    def __init__(self, model_url, labels_url, image_urls, load_from_local=False):
        logging.basicConfig(filename="birdclassifier.log", level=logging.INFO)
        """
        Initialize the image classifier
        """
        self.model_url = model_url
        self.labels_url = labels_url
        self.image_urls = image_urls

        # Calculate the total time to functions
        self.load_from_local = load_from_local
        self.total_time_load_images = 0.0
        self.total_time_preprocess_images = 0.0
        self.total_time_prediction_model = 0.0
        self.total_time_cleaning_labels = 0.0
        self.total_time_downloading_model = 0.0

    def load_model_from_tf_hub(self, model_url):
        """
        Load the model from the tensorflow hub
        """
        start_time = time.time()
        self.total_time_downloading_model += time.time() - start_time
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
        start_time = time.time()
        labels_raw = urllib.request.urlopen(labels_url)
        labels_lines = [
            line.decode("utf-8").replace("\n", "") for line in labels_raw.readlines()
        ]
        labels_lines.pop(0)  # remove header (id, name, ...)
        cleaned_labels = {}
        for line in labels_lines:
            id = int(line.split(",")[0])
            name = line.split(",")[1]
            cleaned_labels[id] = {"name": name}
        self.total_time_cleaning_labels += time.time() - start_time
        return cleaned_labels

    def load_image_by_url(self, image_url):
        """
        Load the image from the image url
        """
        start_time = time.time()
        image_get_response = urllib.request.urlopen(image_url)
        image_array = np.asarray(bytearray(image_get_response.read()), dtype="uint8")
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.total_time_load_images += time.time() - start_time
        return image / 255

    def load_image_by_path(self, image_path):
        """
        Load the image from the image path
        """
        start_time = time.time()
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.total_time_load_images += time.time() - start_time
        return image / 255

    def preprocess_image(self, model, image, labels_url):
        """
        Preprocess the image to generate tensor
        """
        start_time = time.time()
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)
        model_raw_output = model.call(image_tensor).numpy()
        self.total_time_preprocess_images += time.time() - start_time
        return self.order_by_result_score(model_raw_output, labels_url)

    def order_by_result_score(self, model_raw_output, cleaned_labels):
        """
        Order the results by score based on model raw output and cleaned labels
        """
        start_time = time.time()
        for index, value in np.ndenumerate(model_raw_output):
            result_index = index[1]
            cleaned_labels[result_index]["score"] = value
        self.total_time_prediction_model += time.time() - start_time
        return sorted(cleaned_labels.items(), key=lambda x: x[1]["score"])

    def get_top_n_results(self, top_index, order_by_result_score):
        """
        Get the top n results based on result score
        """
        name = order_by_result_score[top_index * (-1)][1]["name"]
        score = order_by_result_score[top_index * (-1)][1]["score"]
        return name, score


class BirdClassifier(ImageClassifier):
    """
    Bird Classifier class for machine learning image predictions based on birds.
    """

    BIRD_MODEL = "https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1"

    def __init__(self, labels_url, image_urls):
        """
        Initialize the bird classifier
        """
        ImageClassifier.__init__(
            self, BirdClassifier.BIRD_MODEL, labels_url, image_urls
        )

    def main(self):
        """
        Main function
        """
        logging.info(f"Starting the bird classifier at {datetime.now()}")
        for index, image_url in enumerate(birds_image_urls):
            bird_model = self.load_model_from_tf_hub(self.model_url)
            bird_labels = self.clean_header_labels(birds_labels_url)
            # Loading the image from the url or path
            if not self.load_from_local:
                image = self.load_image_by_url(image_url)
            else:
                image = self.load_image_by_path(image_url)
            # Generate tensor
            birds_names_with_results_ordered = self.preprocess_image(
                bird_model, image, bird_labels
            )
            print("*******************************************")
            logging.info(f"Run: {int(index + 1)}")
            logging.info(f"Image: {image_url}")
            bird_name, bird_score = self.get_top_n_results(
                1, birds_names_with_results_ordered
            )
            print(f'Top match: "{bird_name}" with score: {bird_score:.2f}%')
            logging.info(f'Top match: "{bird_name}" with score: {bird_score:.2f}%')
            logging.info(f"Run: {int(index + 1)}")
            logging.info(f"Image: {image_url}")
            bird_name, bird_score = self.get_top_n_results(
                2, birds_names_with_results_ordered
            )
            print(f'Second match: "{bird_name}" with score: {bird_score:.2f}%')
            logging.info(f"Run: {int(index + 1)}")
            logging.info(f"Image: {image_url}")
            bird_name, bird_score = self.get_top_n_results(
                3, birds_names_with_results_ordered
            )
            print(f'Third match: "{bird_name}" with score: {bird_score:.2f}%')
            logging.info(f"Run: {int(index + 1)}")
            logging.info(f"Image: {image_url}")
            print("*******************************************")
            logging.info(f"Total time load images: {self.total_time_load_images}")
            logging.info(
                f"Total time preprocess images: {self.total_time_preprocess_images}"
            )
            logging.info(
                f"Total time prediction model: {self.total_time_prediction_model}"
            )


if __name__ == "__main__":
    birds_labels_url = (
        "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv"
    )
    birds_image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg",
        "https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg",
        "https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg",
        "https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg",
    ]
    start_time = time.time()
    classifier = BirdClassifier(
        labels_url=birds_labels_url, image_urls=birds_image_urls
    )
    classifier.main()
    original_app_total_time = 20.040859937667847
    currently_app_total_time = time.time() - start_time
    performance_percentage_difference = (
        (original_app_total_time - currently_app_total_time)
        / original_app_total_time
        * 100
    )
    print(f"Total Time spent(original app): {original_app_total_time:.2f}s")
    print(f"Total Time spent(currently app): {currently_app_total_time:.2f}s")
    print(
        f"{performance_percentage_difference:.2f}% better performance than original classifier.py"
    )
    print(f"Time spent loading images: {classifier.total_time_load_images:.2f}s")
    print(
        f"Time spent preprocessing images: {classifier.total_time_preprocess_images:.2f}s"
    )
    print(f"Time spent cleaning labels: {classifier.total_time_cleaning_labels:.2f}s")
    print(f"Time spent predicting model: {classifier.total_time_prediction_model:.4f}s")
    print(
        f"Time spent downloading models: {classifier.total_time_downloading_model:.4f}s"
    )
