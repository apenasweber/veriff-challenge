import os

path = os.getcwd()

# Mocking model, labels and image urls
test_model_url = f"{path}\\tests/model\\"
test_labels_url = f"{path}\\tests\\labels\\aiy_birds_V1_labelmap.csv"
test_image_urls = [
    f"{path}\\tests\\images\\1.jpg",
    f"{path}\\tests/images\\2.jpg",
    f"{path}\\tests\\images\\3.jpg",
    f"{path}\\tests\\images\\4.jpg",
    f"{path}\\tests\\images\\5.jpg",
]


class TestImageClassifier:
    def test_classifier_model_url_is_valid(self):
        assert os.path.exists(test_model_url)

    def test_classifier_model_url_exists(self):
        assert test_model_url is not None

    def test_classifier_labels_url_exists(self):
        assert test_labels_url is not None

    def test_classifier_image_urls_exists(self):
        assert test_image_urls is not None

    def test_classifier_clean_labels_url_from_csv(self):
        assert test_labels_url.endswith(".csv")

    def test_classifier_image_urls_are_valid(self):
        for url in test_image_urls:
            assert os.path.exists(url)

    def test_classifier_model_url_is_valid(self):
        assert os.path.exists(test_model_url)

    def test_classifier_labels_url_is_valid(self):
        assert os.path.exists(test_labels_url)

    def test_classifier_model_url_is_valid(self):
        assert os.path.exists(test_model_url)

    def test_classifier_labels_url_is_valid(self):
        assert os.path.exists(test_labels_url)

    def test_classifier_model_url_is_valid(self):
        assert os.path.exists(test_model_url)

    def test_classifier_labels_url_is_valid(self):
        assert os.path.exists(test_labels_url)

    def test_classifier_model_url_is_valid(self):
        assert os.path.exists(test_model_url)


class TestBirdClassifier:
    pass
