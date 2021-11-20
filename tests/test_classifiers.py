import pytest, os

path = os.path.dirname(os.path.realpath(__file__))
print(path)

test_model_url = f'{path}/'
test_labels_url = f'{path}labels/aiy_birds_V1_labelmap.csv'
test_image_urls = [
    f'{path}/images/1.jpg',
    f'{path}/images/2.jpg',
    f'{path}/images/3.jpg',
    f'{path}/images/4.jpg',
    f'{path}/images/5.jpg'
]

def test_classifier_model_url_exists():
    assert test_model_url is not None

def test_classifier_labels_url_exists():
    assert test_labels_url is not None

def test_classifier_image_urls_exists():
    assert test_image_urls is not None

def test_classifier_model_url__by_tfhub_is_valid():
    assert test_model_url.startswith('https://tfhub.dev/')

def test_classifier_image_url_is_jpg():
    assert test_image_urls[0].endswith('.jpg')

def test_should_raise_exception_if_model_url_is_not_valid():
    with pytest.raises(ValueError):
        model_url = 'https://www.google.com'

def 


