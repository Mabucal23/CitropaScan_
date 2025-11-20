import tensorflow as tf

# Load the original model
model = tf.keras.models.load_model('accDiseases.keras')

# Create a TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set optimization flag to reduce size (float16 quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Convert the model
tflite_model = converter.convert()

# Save the new, smaller model
with open('accDiseases_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("Quantization to TFLite (float16) complete!")