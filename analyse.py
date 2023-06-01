#!/usr/bin/env python

import flatbuffers
import numpy as np
import sys
sys.path.append("tflite/")
import Model
from tfl_op_names import TFL_OP_NAMES

def load_model_from_file(model_filename):
  with open(model_filename, "rb") as file:
    buffer_data = file.read()
  model_obj = Model.Model.GetRootAsModel(buffer_data, 0)
  model = Model.ModelT.InitFromObj(model_obj)
  return model

def save_model_to_file(model, model_filename):
  builder = flatbuffers.Builder(1024)
  model_offset = model.Pack(builder)
  builder.Finish(model_offset, file_identifier=b'TFL3')
  model_data = builder.Output()
  with open(model_filename, 'wb') as out_file:
    out_file.write(model_data)

def model_memory_stats(model):
  for subgraph in model.subgraphs:
    tensors = subgraph.tensors
    operators = subgraph.operators

    tensor_first_write = [-1] * len(tensors)
    for input in subgraph.inputs:
      tensor_first_write[input] = 0
    for step, operator in enumerate(operators):
      for output in operator.outputs:
        if tensor_first_write[output] == -1:
          tensor_first_write[output] = step

    tensor_last_read = [-1] * len(tensors)
    for output in subgraph.outputs:
      tensor_last_read[output] = len(operators) - 1
    for step, operator in reversed(list(enumerate(operators))):
      for input in operator.inputs:
        if tensor_last_read[input] == -1:
          tensor_last_read[input] = step

    result = []
    for step, operator in enumerate(operators):
      total_memory = 0
      op_code = model.operatorCodes[operator.opcodeIndex].builtinCode
      op_name = TFL_OP_NAMES[op_code]
      for tensor_index, tensor in enumerate(tensors):
        first_write = tensor_first_write[tensor_index]
        last_read = tensor_last_read[tensor_index]
        if first_write == -1 or last_read == -1:
          continue
        if step < first_write or step > last_read:
          continue
        shape = tensor.shape
        element_count = 1
        for dim in shape:
          element_count *= dim
        total_memory += element_count
      result.append((op_name, total_memory))
  return result

def print_memory_stats(memory_stats):
  high_water_mark = 0
  for step, stats in enumerate(memory_stats):
      op_name = stats[0]
      total_memory = stats[1]
      if total_memory > high_water_mark:
        high_water_mark = total_memory
      print("%d\t%s\t%s" % (step, op_name, f'{total_memory:,}'))
  print("Minimum RAM required is", f'{high_water_mark:,}', "bytes")

if __name__ == '__main__':

  model_filename = sys.argv[1]

  model = load_model_from_file(model_filename)

  mem_stats = model_memory_stats(model)

  print_memory_stats(mem_stats)