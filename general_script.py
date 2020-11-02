import tensorflow as tf
import time
import numpy as np
import os

# Asks for needed user inputs
model_path = input("Enter the path for the tensorflow model: ")
data_path = input("Enter the path for the dataset: ")
labels_path = input("Enter the path for the labels.txt file: ")
quant_path = input("Enter the path for the folder to save the quantized model to: ") + "model_quant.tflite"
data_type = input("Enter the data type out want to use ('1' for  Int8, '2' for Int16, and '3' for Float16")
number_of_batches = input("How many batches of data do you want to test the models on? (1 Batch is 32 images)")


# Load the dataset
batch_size = 32
img_height = 224
img_width = 224
test_images_ds = tf.keras.preprocessing.image_dataset_from_directory(data_path, image_size=(img_height, img_width), batch_size=batch_size, shuffle=False, labels="inferred")
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
test_images_ds_map = test_images_ds.map(lambda x, y: (normalization_layer(x), y))

# creates a dictionary for the class number to folder name
number_to_folder_dict = {}
subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]
i = 0
for path in subfolders:
    split = os.path.split(path)
    name = split[1]
    number_to_folder_dict[i] = name
    i = i + 1

# creates a dictionary for the folder name to model output label
folder_to_label_dict = {}
with open(labels_path) as f:
    for line in f:
        splits = line.split("\t")
        key = splits[0]
        length = len(splits[1])
        val = splits[1]
        val = val[0:length - 1]
        folder_to_label_dict[key] = val

# Create Test Labels from the dataset
test_labels = []

for images, labels in test_images_ds_map.take(number_of_batches):
    for label in labels:
        folder = number_to_folder_dict[label.numpy()]
        label = folder_to_label_dict[folder]
        test_labels.append(label)


# Quantisize the model and save it to a folder
def representative_data_gen():
  for input_value in test_images_ds_map.take(5):
    yield [input_value]


converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

if data_type == "1":
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    converter.representative_dataset = representative_data_gen
elif data_type == "2":
    converter.target_spec.supported_ops = [tf.uint16]
    converter.inference_input_type = tf.uint16
    converter.inference_output_type = tf.uint16

    converter.representative_dataset = representative_data_gen
else:
    converter.target_spec.supported_ops = [tf.float16]
    converter.inference_input_type = tf.float16
    converter.inference_output_type = tf.float16

tflite_model_quant = converter.convert()
with tf.io.gfile.GFile(quant_path, 'wb') as f:
    f.write(tflite_model_quant)


# Load the models into interpreters
interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()

interpreter_quant = tf.lite.Interpreter(model_path=str(quant_path))
interpreter_quant.allocate_tensors()


# Evaluating the Models
# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter, quant_model):
    # Get the input and output details of the model
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Start the timer
    start_time = time.time()

    # Run predictions for the specified number of batches from the dataset.
    prediction_digits = []
    for images, labels in test_images_ds_map.take(number_of_batches):
        for test_image in images:
            # Pre-processing: add batch dimension and convert to float32 to match with
            # the model's input data format.
            if not quant_model:
                test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
            else:
                test_image = np.expand_dims(test_image, axis=0)
                if data_type == "1":
                    test_image = tf.image.convert_image_dtype(test_image, np.uint8)
                elif data_type == "2":
                    test_image = tf.image.convert_image_dtype(test_image, np.uint16)
                else:
                    test_image = tf.image.convert_image_dtype(test_image, np.float16)

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


print(evaluate_model(interpreter, False))
print(evaluate_model(interpreter_quant, True))
