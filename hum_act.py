import streamlit as st
import numpy as np
import base64
from tensorflow.keras.models import load_model
import random
import cv2
model2 = load_model("face_action.h5")

def preprocess_image(image):
    new_width, new_height = 180, 180
    resized_image = cv2.resize(image, (new_width, new_height))
    return np.expand_dims(resized_image, axis=0)

def get_top_class_label(predictions):
    new_labels = ["happy", "sad", "surprise"]
    top_class_index = np.argmax(predictions[0])
    top_class_label = new_labels[top_class_index]
    return top_class_label


happy_phrases = [
    "You seem to be very happy!",
    "Your happiness is contagious!",
    "I sense a lot of joy in your expression.",
    "Your smile is shining brightly!",
    "Your happiness is radiating!",
    "Seeing your joy brings a smile to my face.",
    "I can tell you're in a great mood!",
    "You're beaming with happiness!",
    "Your positive energy is palpable.",
    "Your happiness is a ray of sunshine!"
]

sad_phrases = [
    "Difficult roads often lead to beautiful destinations.",
    "You are stronger than you know, and braver than you think.",
    "Challenges are what make life interesting, and overcoming them is what makes life meaningful.",
    "Every day may not be good, but there's something good in every day.",
    "Believe in yourself and all that you are. Know that there's something inside you that is greater than any obstacle.",
    "You're not defined by your struggles, but by your ability to overcome them.",
    "Don't let a tough day make you forget how far you've come.",
    "The sun will rise and we will try again. Every new day is a chance to begin again.",
    "Success is not final, failure is not fatal: It is the courage to continue that counts.",
    "Keep going. You've survived 100% of your worst days so far."
]

surprise_phrases = [
    "What a surprise! Your expression caught me off guard.",
    "You've managed to surprise me with your look!",
    "It seems like something unexpected just happened.",
    "I can see the shock on your face!",
    "Your surprise is palpable!",
    "This unexpected moment is quite intriguing.",
    "Life has a way of surprising us, doesn't it?",
    "You've certainly grabbed my attention!",
    "The element of surprise adds excitement to life.",
    "Who could have predicted this surprise?"
]

def main():
    st.title("Human Behavior Recognition using CNN")

    # Display the problem statement
    st.markdown("<h3>Problem Statement:</h3>", unsafe_allow_html=True)
    st.write("The goal of this project is to develop an AI model that employs Convolutional Neural Networks (CNN) for accurate recognition of human behaviors depicted in images.")

    # Display the objectives
    st.markdown("<h3>Objectives:</h3>", unsafe_allow_html=True)
    st.write("1. Construct a CNN-based model capable of precisely classifying a wide range of human behaviors based on input images.")
    st.write("2. Assess the model's performance using pertinent metrics to ensure its efficacy.")
    st.write("3. Furnish users with reliable predictions for uploaded images, accompanied by associated confidence scores.")
    st.write("4. Enhance user experience by delivering informative and motivational messages aligned with the predicted behavior class.")

    st.subheader("images Represented:")
    st.write("1.Happy Face.")
    image_url1 = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQWzzxYjT8j698i-mG41XSj4V2W3vdrGnKy9Q&usqp=CAU"
    st.image(image_url1)
    st.write("2. Sad Face.")
    image_url2 = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT7IZbZtJw39MpV_a9oWEKGPEAQQZwjUsl97w&usqp=CAU"
    st.image(image_url2)
    st.write("3. surprised Face.")
    image_url3 = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTb1we3Yccu16JgBEFunZ-pNxB25xuk9Zwt8A&usqp=CAU"
    st.image(image_url3)

    st.subheader("Upload your image to determine the class to which it belongs.")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = uploaded_image.read()
        image_array = np.frombuffer(image, dtype=np.uint8)
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        processed_image = preprocess_image(decoded_image)
        prediction = model2.predict(processed_image)

        top_class_label = get_top_class_label(prediction)

        st.image(decoded_image, caption="Uploaded Image", use_column_width=True)

        if top_class_label == "happy":
            selected_phrase = random.choice(happy_phrases)
        elif top_class_label == "sad":
            selected_phrase = random.choice(sad_phrases)
        elif top_class_label == "surprise":
            selected_phrase = random.choice(surprise_phrases)

        st.subheader(f"I think you are so {top_class_label}!")
        st.subheader(selected_phrase)



    st.markdown("<h3>Disclaimer:</h3>", unsafe_allow_html=True)
    st.write("The predictions provided by our AI model on this web page are based on a Convolutional Neural Network (CNN) trained on a limited dataset of human behavior images resized to 144 pixels due to resource constraints. While we have endeavored to create an accurate model, it is essential to acknowledge the following potential limitations:")
    
    st.write("1. **Reduced Resolution:** The images used for training were resized to 144 pixels, which may result in reduced model performance compared to higher-resolution models.")
    st.write("2. **Resource Constraints:** Due to resource limitations, the training dataset may not encompass all possible variations and complexities of human behaviors, which could affect the model's accuracy.")
    st.write("3. **Mixed Behaviors:** Some images in the dataset might depict a mixture of multiple behaviors, making it challenging for the model to provide precise predictions for such cases.")
    st.write("4. **Ambiguity in Predictions:** In scenarios where the model encounters images with mixed behaviors or ambiguous features, the predictions might be uncertain or incorrect.")
    st.write("5. **No 100% Accuracy Guarantee:** Achieving 100% accuracy in human behavior recognition, especially with mixed-behavior images, is a challenging task, and our model may not achieve perfection in all cases.")

    st.write("While our AI model aims to provide useful predictions, it may not always be flawless due to these inherent limitations. We encourage users to interpret the predictions as informative and for entertainment purposes. The model's performance is subject to the constraints of the underlying technology and the training data. Your feedback and understanding are invaluable as we continuously strive to enhance our AI system. Thank you for using our service!")



if __name__ == "__main__":
    main()
