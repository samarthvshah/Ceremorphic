import tensorflow as tf
import numpy as np
import time
import os
import pickle
import matplotlib.pylab as plt

# File Paths for Both Normal and Quantized Models
model_path_quant = "saved_models/tflite/mobilenet_v1_1.0_224_quant.tflite"
model_path_normal = "saved_models/tflite/mobilenet_v1_1.0_224.tflite"
images_path = "data/mobilenet/imagenet-mini/val"
syn_to_words_path = "data/mobilenet/syn_to_words.txt"
words_to_number_path = "data/mobilenet/imagenet1000_clsid_to_human.pkl.txt"

# Loading Data
batch_size = 32
img_height = 224
img_width = 224

test_labels = []
test_images = []


# creates a dictionary for the synset key to the names
syn_to_name_dict = {}
with open(syn_to_words_path) as f:
    for line in f:
        splits = line.split("\t")
        key = splits[0]
        length = len(splits[1])
        val = splits[1]
        val = val[0:length - 1]
        syn_to_name_dict[key] = val

# creates a dictionary for the name to class number
name_to_number_dict = {}
file = open(words_to_number_path, 'rb')
object_file = pickle.load(file)
i = 0
for element in object_file:
    name_to_number_dict[object_file[i]] = i
    i = i + 1
file.close()

# creates a dictionary for the label to the synset key
folder_to_syn_dict = {}
subfolders = [f.path for f in os.scandir(images_path) if f.is_dir()]
i = 0
for path in subfolders:
    name = path[33:44]
    folder_to_syn_dict[i] = name
    i = i + 1

test_images_ds = tf.keras.preprocessing.image_dataset_from_directory(images_path, image_size=(img_height, img_width), batch_size=batch_size, shuffle=False, labels="inferred")
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
test_images_ds_map = test_images_ds.map(lambda x, y: (normalization_layer(x), y))

for images, labels in test_images_ds_map.take(5):
    for label in labels:
        syn = folder_to_syn_dict[label.numpy()]
        words = syn_to_name_dict[syn]
        number = name_to_number_dict[words]
        test_labels.append(number)
    first_image = images[0].numpy()


print("starting eval")

# Load the models into interpreters
interpreter = tf.lite.Interpreter(model_path=str(model_path_normal))
interpreter.allocate_tensors()

interpreter_quant = tf.lite.Interpreter(model_path=str(model_path_quant))
interpreter_quant.allocate_tensors()


# input_index_quant = interpreter_quant.get_input_details()[0]["index"]
# output_index_quant = interpreter_quant.get_output_details()[0]["index"]


# # Testing the regular model on 1 image
# test_image = np.expand_dims(first_image, axis=0)
#
# input_index = interpreter.get_input_details()[0]["index"]
# output_index = interpreter.get_output_details()[0]["index"]
# interpreter.set_tensor(input_index, test_image)
# interpreter.invoke()
# predictions = interpreter.get_tensor(output_index)
#
# plt.imshow(first_image)
# template = "True:{true}, predicted:{predict}"
# _ = plt.title(template.format(true=str(test_labels[0]), predict=str(np.argmax(predictions[0])-1)))
# plt.grid(False)
# plt.show()
#
# # Testing the quant model on 1 image
# test_image_int = np.expand_dims(first_image, axis=0)
# test_image_int = tf.image.convert_image_dtype(test_image_int, np.uint8)
#
# input_index = interpreter_quant.get_input_details()[0]["index"]
# output_index = interpreter_quant.get_output_details()[0]["index"]
# interpreter_quant.set_tensor(input_index, test_image_int)
# interpreter_quant.invoke()
# predictions = interpreter_quant.get_tensor(output_index)
#
# plt.imshow(first_image)
# template = "True:{true}, predicted:{predict}"
# _ = plt.title(template.format(true=str(test_labels[0]), predict=str(np.argmax(predictions[0])-1)))
# plt.grid(False)
# plt.show()

# Evaluating the Models

# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter, is_float):
    start_time = time.time()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    for images, labels in test_images_ds_map.take(5):
        for test_image in images:
            # Pre-processing: add batch dimension and convert to float32 to match with
            # the model's input data format.
            if is_float:
                test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
            else:
                test_image = np.expand_dims(test_image, axis=0)
                test_image = tf.image.convert_image_dtype(test_image, np.uint8)

            interpreter.set_tensor(input_index, test_image)

            # Run inference.
            interpreter.invoke()

            # Post-processing: remove batch dimension and find the digit with highest
            # probability.
            output = interpreter.tensor(output_index)
            digit = np.argmax(output()[0])
            prediction_digits.append(digit-1)

    # Compare prediction results with ground truth labels to calculate accuracy.
    accurate_count = 0
    for index in range(len(prediction_digits)):
        if prediction_digits[index] == test_labels[index]:
            accurate_count += 1
    accuracy = accurate_count * 1.0 / len(prediction_digits)

    print("--- %s seconds ---" % (time.time() - start_time))
    return accuracy


print(evaluate_model(interpreter_quant, False))
print(evaluate_model(interpreter, True))

