import tensorflow as tf

model = tf.keras.models.load_model('mobilenet_7.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('fer_mobilenet.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted successfully!")