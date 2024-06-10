# import numpy as np
# import tensorflow as tf
# from PIL import Image
# import cv2
#
# IMG_WIDTH = 30  # Set the image width used during training
# IMG_HEIGHT = 30  # Set the image height used during training
# NUM_CATEGORIES = 43  # Set the number of categories based on the training dataset
# MODEL_PATH = 'traffic_sign_model.h5'  # Path to the saved model
#
# def main():
#     # Load the trained model
#     model = tf.keras.models.load_model(MODEL_PATH)
#
#     # Open a connection to the camera
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise Exception("Could not open video device")
#
#     try:
#         while True:
#             # Capture frame-by-frame
#             ret, frame = cap.read()
#             if not ret:
#                 print("Failed to capture image")
#                 break
#
#             # Preprocess the frame
#             image_array = preprocess_frame(frame)
#
#             # Predict the traffic sign category
#             prediction = predict(model, image_array)
#
#             # Display the prediction on the frame
#             cv2.putText(frame, f"Prediction: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#
#             # Display the resulting frame
#             cv2.imshow('Traffic Sign Recognition', frame)
#
#             # Break the loop if 'q' key is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         # Release the camera and close all OpenCV windows
#         cap.release()
#         cv2.destroyAllWindows()
#
# def preprocess_frame(frame):
#     """
#     Preprocess the captured frame to the format expected by the model.
#     """
#     try:
#         img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
#         img_array = np.array(img)
#         if img_array.shape == (IMG_WIDTH, IMG_HEIGHT, 3):
#             img_array = np.expand_dims(img_array, axis=0)
#             img_array = img_array / 255.0  # Normalize the image
#             return img_array
#         else:
#             raise ValueError("Frame does not have 3 channels")
#     except Exception as e:
#         raise ValueError(f"Error processing frame: {e}")
#
# def predict(model, image_array):
#     """
#     Predict the category of the traffic sign using the trained model.
#     """
#     predictions = model.predict(image_array)
#     predicted_category = np.argmax(predictions[0])
#     return predicted_category
#
# if __name__ == "__main__":
#     main()

import sys
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_WIDTH = 30  # Set the image width used during training
IMG_HEIGHT = 30  # Set the image height used during training
NUM_CATEGORIES = 43  # Set the number of categories based on the training dataset
MODEL_PATH = 'traffic_sign_model.h5'  # Path to the saved model

def main():
    # Hardcode the image path
    image_path = 'gtsrb/40/00000_00000.ppm'

    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Preprocess the input image
    image_array = preprocess_image(image_path)

    # Predict the traffic sign category
    prediction = predict(model, image_array)

    # Print the predicted category
    print(f"Predicted category: {prediction}")

def preprocess_image(image_path):
    """
    Preprocess the input image to the format expected by the model.
    """
    try:
        with Image.open(image_path) as img:
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = np.array(img)
            if img_array.shape == (IMG_WIDTH, IMG_HEIGHT, 3):
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # Normalize the image
                return img_array
            else:
                raise ValueError(f"Image does not have 3 channels: {image_path}")
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")

def predict(model, image_array):
    """
    Predict the category of the traffic sign using the trained model.
    """
    predictions = model.predict(image_array)
    predicted_category = np.argmax(predictions[0])
    return predicted_category

if __name__ == "__main__":
    main()
