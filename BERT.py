from nltk.tbl import feature
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import albert
from albert import squad_utils as sq
import pandas

# Paths
model_path = "saved_models/tensorflow/albert"
tflite_model_path = 'saved_models/tflite/albert_lite_base_squadv1_1.tflite'
quant_model_path = 'saved_models/tflite/albert_model_quant.tflite'
data_path = 'data/albert/train-v2.0.json'


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

examples = sq.read_squad_examples(data_path, True)
tokenizer = sq.tokenization.WordpieceTokenizer
features = sq.convert_examples_to_features(examples, tokenizer, 384)
feature = features[1]


# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_ids = feature.input_ids
input_mask = feature.input_mask
segment_ids = feature.segment.ids

input_ids = np.array(input_ids, dtype=np.int32)
input_mask = np.array(input_mask, dtype=np.int32)
segment_ids = np.array(segment_ids, dtype=np.int32)

interpreter.set_tensor(input_details[0]["index"], input_ids)
interpreter.set_tensor(input_details[1]["index"], input_mask)
interpreter.set_tensor(input_details[2]["index"], segment_ids)
interpreter.invoke()

# Get output logits.
end_logits = interpreter.get_tensor(output_details[0]["index"])[0]
start_logits = interpreter.get_tensor(output_details[1]["index"])[0]

print(start_logits)
print(end_logits)