import tensorflow as tf 
from tensorflow.keras.models import load_model
import os

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "art_style_recognition_final_model.h5")
MODEL = load_model(MODEL_FILE)
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ["Abstract", "Baroque", "Color Field Painting", "Cubism", "Expressionism", "Fauvism", "Impressionism", "Minimalism", "Pointillism", "Pop", "Realism", "Renaissance", "Romanticism", "Ukiyo"]
TOPK = 5

def predict_category(image_path):
    
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))    
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)

    predictions = MODEL.predict(img)

    top5_predictions = tf.math.top_k(predictions, k=TOPK)

    top5_indices = top5_predictions.indices.numpy()
    top5_probability = top5_predictions.values.numpy()

    result_top5_categories = []
    result_top5_probability = []
    for i in range(TOPK):
        result_top5_categories.append(CLASS_NAMES[top5_indices[0][i]])
        result_top5_probability.append(round(top5_probability[0][i] * 100, 2))
    
    return result_top5_categories, result_top5_probability