import logging
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import matplotlib.pylab as plt
import time
# from tensorflow.keras import layersi

logging.getLogger("tensorflow").setLevel(logging.DEBUG)


# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28)),
    keras.layers.Reshape(target_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
])

# Train the digit classification model
model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))

# Convert the model to TfLite and save it to a file
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_models_dir = pathlib.Path("saved_models/tflite/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir / "mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)

# Quantisize the tflite model and save it to another file
converter.optimizations = [tf.lite.Optimize.DEFAULT]

mnist_train, _ = tf.keras.datasets.mnist.load_data()
images = tf.cast(mnist_train[0], tf.float32) / 255.0
mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)




def representative_data_gen():
    for input_value in mnist_ds.take(100):
        # Model has only one input so each data point has one element.
        # input_value = tf.random.uniform(
        #     (1, 28, 28), minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
        # )
        yield [input_value]



converter.representative_dataset = representative_data_gen

tflite_model_quant = converter.convert()
tflite_model_quant_file = tflite_models_dir / "mnist_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)

# Load the models into interpreters
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
interpreter_quant.allocate_tensors()
input_index_quant = interpreter_quant.get_input_details()[0]["index"]
output_index_quant = interpreter_quant.get_output_details()[0]["index"]

# Testing the regular model on 1 image
test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
interpreter.set_tensor(input_index, test_image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_index)

plt.imshow(test_images[0])
template = "True:{true}, predicted:{predict}"
_ = plt.title(template.format(true=str(test_labels[0]), predict=str(np.argmax(predictions[0]))))
plt.grid(False)

# Testing the quant model on 1 image
input_index = interpreter_quant.get_input_details()[0]["index"]
output_index = interpreter_quant.get_output_details()[0]["index"]
interpreter_quant.set_tensor(input_index, test_image)
interpreter_quant.invoke()
predictions = interpreter_quant.get_tensor(output_index)

plt.imshow(test_images[0])
template = "True:{true}, predicted:{predict}"
_ = plt.title(template.format(true=str(test_labels[0]), predict=str(np.argmax(predictions[0]))))
plt.grid(False)


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
