import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model(r"C:\Users\arpan\car_parking_detection_space_count\model_final.h5")

# Convert to TFLite with FP16 optimization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save
with open("model_final.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite model saved!")