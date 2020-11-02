import tensorflow as tf
import numpy as np
import time


from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# LOAD AND SPLIT DATASET and normalize pixel values
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0


# Define the paths for the files to be saved at
converted_model_path = 'saved_models/tflite/cifar10_model.tflite'
quant_model_path = 'saved_models/tflite/cifar10_model_quant.tflite'

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model('saved_models/tensorflow/cifar10')
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile(converted_model_path, 'wb') as f:
    f.write(tflite_model)

# Quantize the model and save it into a new file
converter.optimizations = [tf.lite.Optimize.DEFAULT]
images_train, _ = datasets.cifar10.load_data()
images = tf.cast(images_train[0], tf.float32) / 255
mnist_ds = tf.data.Dataset.from_tensor_slices(images).batch(1)

# Generates the Representative Data Set from the first 100 images
def representative_data_gen():
    for input_value in mnist_ds.take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]


converter.representative_dataset = representative_data_gen

tflite_model_quant = converter.convert()
with tf.io.gfile.GFile(quant_model_path, 'wb') as f:
    f.write(tflite_model_quant)

# Load the models into interpreters
interpreter = tf.lite.Interpreter(model_path=str(converted_model_path))
interpreter.allocate_tensors()

interpreter_quant = tf.lite.Interpreter(model_path=str(quant_model_path))
interpreter_quant.allocate_tensors()
input_index_quant = interpreter_quant.get_input_details()[0]["index"]
output_index_quant = interpreter_quant.get_output_details()[0]["index"]

# Testing the regular model on 1 image
test_image = np.expand_dims(test_images[1], axis=0).astype(np.float32)

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
interpreter.set_tensor(input_index, test_image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_index)

plt.imshow(test_images[1])
template = "True:{true}, predicted:{predict}"
_ = plt.title(template.format(true=str(test_labels[1]), predict=str(np.argmax(predictions[0]))))
plt.grid(False)
plt.show()

# Testing the quant model on 1 image
input_index = interpreter_quant.get_input_details()[0]["index"]
output_index = interpreter_quant.get_output_details()[0]["index"]
interpreter_quant.set_tensor(input_index, test_image)
interpreter_quant.invoke()
predictions = interpreter_quant.get_tensor(output_index)

plt.imshow(test_images[1])
template = "True:{true}, predicted:{predict}"
_ = plt.title(template.format(true=str(test_labels[1]), predict=str(np.argmax(predictions[0]))))
plt.grid(False)
plt.show()


# Evaluating the Models
# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter):
    start_time = time.time()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    for test_image in test_images:
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    # Compare prediction results with ground truth labels to calculate accuracy.
    accurate_count = 0
    for index in range(len(prediction_digits)):
        if prediction_digits[index] == test_labels[index]:
            accurate_count += 1
    accuracy = accurate_count * 1.0 / len(prediction_digits)

    print("--- %s seconds ---" % (time.time() - start_time))
    return accuracy


print(evaluate_model(interpreter))
print(evaluate_model(interpreter_quant))
