import tensorflow as tf
import h5py

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="saved_models/tflite/cifar10_model_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# get details for each layer
all_layers_details = interpreter.get_tensor_details()

for layer in all_layers_details:
    print(layer["index"])
    print(layer['name'])
    print(layer['shape'])
    print(layer['dtype'])
    print(layer['quantization'])
    try:
        print(interpreter.get_tensor(layer["index"]))
    except ValueError:
        print("No weights for this layer")
    print()
    print()
