import tensorflow as tf
from tensorflow.keras.models import load_model


loaded_model = load_model('art_style_recognition_final_model.h5')

IMG_HEIGHT = 224 # VGG's dim
IMG_WIDTH = 224 # VGG's dim

# Function to predict categories for an image file
def predict_categories(image_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)  # Adjust preprocessing as needed for your model

    # Make predictions
    predictions = loaded_model.predict(img)

    # Get the top 5 predicted categories and their probabilities
    top5_indices = tf.math.top_k(predictions, k=5).indices.numpy()
    top5_probabilities = tf.math.top_k(predictions, k=5).values.numpy()

    # Map indices to class names (assuming you have a list of class names)
    class_names = ["Abstract", "Baroque", "Color Field Painting", "Cubism", "Expressionism", "Fauvism", "Impressionism", "Minimalism", "Pointillism", "Pop", "Realism", "Renaissance", "Romanticism", "Ukiyo"]  # Replace with your class names
    print(top5_indices)
    top5_categories = []
    for i in range(5):
        top5_categories.append(class_names[top5_indices[0][i]])
    
    return top5_categories, top5_probabilities[0]

image_path = 'test.jpg'
top5_categories, top5_probabilities = predict_categories(image_path)

for i in range(5):
    print(f"Category: {top5_categories[i]}, Probability: {round(top5_probabilities[i] * 100, 2)}%")

