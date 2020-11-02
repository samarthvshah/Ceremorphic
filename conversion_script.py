import tensorflow as tf
import numpy as np

# Asks for needed user inputs
model_path = input("Enter the path for the tensorflow model: ")
data_path = input("Enter the path for the dataset: ")
quant_path = input("Enter the path for the folder to save the quantized model to: ") + "model_quant.tflite"

# Load the dataset
batch_size = 20
img_height = 224
img_width = 224
test_images_ds = tf.keras.preprocessing.image_dataset_from_directory(data_path, image_size=(img_height, img_width), batch_size=batch_size, shuffle=False, labels="inferred")
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
test_images_ds_map = test_images_ds.map(lambda x, y: (normalization_layer(x), y))

# Quantize the model and save it to a folder
def representative_data_gen():
    for input_value in test_images_ds_map.take(5):
        yield [input_value]


# Load the model into the converter
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

converter.representative_dataset = representative_data_gen

tflite_model_quant = converter.convert()
with tf.io.gfile.GFile(quant_path, 'wb') as f:
    f.write(tflite_model_quant)