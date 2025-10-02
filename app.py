import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
from io import BytesIO
from PIL import Image

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=r"C:/Users/arpan/car_parking_detection_space_count/model_final.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Load parking positions
@st.cache_data
def load_positions():
    with open(r"C:/Users/arpan/car_parking_detection_space_count/carposition.pkl", "rb") as f:
        return pickle.load(f)

# Process a single frame or image
def process_frame(frame, interpreter, pos_list):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    space_counter = 0
    img = frame.copy()

    for pos in pos_list:
        x, y = pos
        # Crop ROI (130x65 as in main.py)
        roi = img[y:y+65, x:x+130]
        if roi.shape[0] != 65 or roi.shape[1] != 130:
            continue  # Skip invalid ROIs
        # Preprocess for model
        roi_resized = cv2.resize(roi, (48, 48))
        roi_normalized = roi_resized.astype(np.float32) / 255.0
        roi_input = np.expand_dims(roi_normalized, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]["index"], roi_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        class_id = np.argmax(output[0])
        label = "no_car" if class_id == 0 else "car"

        # Count free spaces
        if label == "no_car":
            space_counter += 1
            color = (0, 255, 0)  # Green
        else:
            color = (0, 0, 255)  # Red

        # Draw rectangle and label
        cv2.rectangle(img, (x, y), (x+130, y+65), color, 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img, space_counter

# Streamlit app
def main():
    st.title("Parking Space Detection")
    st.write("Detects cars in parking spaces and counts free spots. Works on mobile browsers!")

    # Load model and positions
    interpreter = load_model()
    pos_list = load_positions()

    # Option to use webcam, uploaded video, or image
    option = st.selectbox("Choose input source:", ["Uploaded Image", "Uploaded Video", "Webcam"])

    if option == "Uploaded Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            frame = np.array(image)
            if frame.shape[-1] == 4:  # Remove alpha channel if present
                frame = frame[:, :, :3]
            # Process
            processed_frame, free_spaces = process_frame(frame, interpreter, pos_list)
            # Display
            st.image(processed_frame, caption=f"Free Spaces: {free_spaces}", use_column_width=True)

    elif option == "Uploaded Video":
        uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
        if uploaded_file is not None:
            # Save video temporarily
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.read())
            cap = cv2.VideoCapture("temp_video.mp4")
            stframe = st.empty()
            space_text = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Process frame
                processed_frame, free_spaces = process_frame(frame, interpreter, pos_list)
                # Convert BGR to RGB
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                # Display
                stframe.image(processed_frame, use_column_width=True)
                space_text.write(f"Free Spaces: {free_spaces}")
            cap.release()

    elif option == "Webcam":
        st.warning("Webcam support is limited in Streamlit. Use a local video or image for best results.")
        # Placeholder for webcam (Streamlit's webrtc is experimental)
        run = st.checkbox("Start Webcam")
        if run:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            space_text = st.empty()
            while run:
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame, free_spaces = process_frame(frame, interpreter, pos_list)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(processed_frame, use_column_width=True)
                space_text.write(f"Free Spaces: {free_spaces}")
                run = st.checkbox("Start Webcam", value=True)
            cap.release()

if __name__ == "__main__":
    main()