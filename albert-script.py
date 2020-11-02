import tensorflow as tf
import tensorflow_hub as hub
import pickle
import numpy as np

model_path = 'saved_models/tensorflow/albert'
quant_path = 'saved_models/tflite/albert_quant.tflite'


converter = tf.lite.TFLiteConverter.from_saved_model(model_path, tags=set({'train'}), signature_keys={"tokens"})


class features_inputs(object):
  def __init__(self, input_ids, input_mask, segment_ids):
    self.input_ids= input_ids
    self.input_mask= input_mask
    self.segment_ids= segment_ids


with open('data/albert/features.txt', 'rb') as fin:
  inputs = pickle.load(fin)


def representative_data_gen():
  for feature in inputs:
    input_ids = feature.input_ids
    input_mask = feature.input_mask
    segment_ids = feature.segment_ids

    input_ids = np.expand_dims(np.array(input_ids, dtype=np.int32), axis=0)
    input_mask = np.expand_dims(np.array(input_mask, dtype=np.int32), axis=0)
    segment_ids = np.expand_dims(np.array(segment_ids, dtype=np.int32), axis=0)

    yield [input_mask, segment_ids, input_ids]


converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_data_gen
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8

model = converter.convert()
with tf.io.gfile.GFile(quant_path, 'wb') as f:
  f.write(model)


interpreter = tf.lite.Interpreter(model_path=quant_path)
interpreter.allocate_tensors()

print(interpreter.get_input_details())
print(interpreter.get_output_details())