import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="RecycleVision",
    page_icon="♻️",
    layout="wide"
)


# ==========================================================
# CLASS NAMES
# ==========================================================

# These must match the classes used during model training
class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# ==========================================================
# LOAD TRAINED RESNET50 MODEL
# ==========================================================

@st.cache_resource
def load_model():
    # Path to the trained ResNet50 model
    model_path = r"C:\Users\HP\Downloads\resnet50_model.keras"

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    return model


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("♻️ RecycleVision")

st.sidebar.write("Garbage Image Classification")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Introduction",
        "Model Prediction"
    ]
)


# ==========================================================
# INTRODUCTION PAGE
# ==========================================================

if page == "Introduction":

    st.title("♻️ RecycleVision")

    st.subheader("See It. Sort It. Recycle It.")

    st.write(
        """
        **RecycleVision** is a deep learning-based garbage image
        classification system that identifies different types of
        waste from an image.
        """
    )

    st.header("🌱 What Does RecycleVision Do?")


    st.write("The system can identify the following six categories:")

    st.markdown(
        """
        - 📦 **Cardboard**
        - 🥛 **Glass**
        - 🔩 **Metal**
        - 📄 **Paper**
        - 🧴 **Plastic**
        - 🗑️ **Trash**
        """
    )

    st.header("🎯 Project Objective")

    st.write(
        """
        The main objective of RecycleVision is to demonstrate how
        deep learning can help automate waste classification and
        support proper waste segregation.
        """
    )

    st.header("🤖 Model Used")

    st.write(
        """
        The final garbage classification model is based on
        **ResNet50 Transfer Learning**.
        
        The pretrained ResNet50 model was adapted and trained
        to classify garbage into six different categories.
        """
    )

    st.header("🔍 How It Works")

    st.markdown(
        """
        **Step 1:** Upload a garbage image.

        **Step 2:** The ResNet50 model analyzes the image.

        **Step 3:** The model predicts the garbage category.

        **Step 4:** RecycleVision displays the predicted class
        and confidence score.
        """
    )

    st.success("♻️ See It. Sort It. Recycle It.")


# ==========================================================
# MODEL PREDICTION PAGE
# ==========================================================

elif page == "Model Prediction":

    st.title("🔍 Garbage Classification")

    st.write(
        """
        Upload an image of a garbage item and RecycleVision
        will predict its waste category.
        """
    )

    # Image uploader
    uploaded_file = st.file_uploader(
        "📤 Upload a Garbage Image",
        type=["jpg", "jpeg", "png"]
    )

    # Check whether an image was uploaded
    if uploaded_file is not None:

        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Display uploaded image
        st.image(
            image,
            caption="Uploaded Image",
            width=400
        )

        # Load the trained model
        model = load_model()

        # Resize image to model input size
        image = image.resize((224, 224))

        # Convert image to NumPy array
        image_array = np.array(image)

        # Make sure image has 3 RGB channels
        if image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Convert pixel values to float
        image_array = image_array.astype("float32")

        # ResNet50 preprocessing
        image_array = tf.keras.applications.resnet50.preprocess_input(
            image_array
        )

        # Make prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )

        # Get predicted class index
        predicted_index = np.argmax(predictions[0])

        # Get predicted class
        predicted_class = class_names[predicted_index]

        # Get confidence
        confidence = predictions[0][predicted_index] * 100

        # Display prediction
        st.success(
            f"Predicted Garbage Class: {predicted_class.capitalize()}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )