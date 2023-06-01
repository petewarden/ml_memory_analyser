import flatbuffers
import sys
import numpy as np

sys.path.append("tflite/")
from Buffer import *
from BuiltinOperator import *
from Tensor import *
from TensorType import *
from SubGraph import *
from OperatorCode import *
from Operator import *
from Model import *

from analyse import *

TFLITE_SCHEMA_VERSION = 3


def flatbuffer_model_from_def(model_def):
    tensors = model_def["tensors"]
    inputs = model_def["inputs"]
    outputs = model_def["outputs"]
    ops = model_def["ops"]

    builder = flatbuffers.Builder(1024)

    BufferStart(builder)
    buffer0_offset = BufferEnd(builder)

    ModelStartBuffersVector(builder, 1)
    builder.PrependUOffsetTRelative(buffer0_offset)
    buffers_offset = builder.EndVector(1)

    tensor_offsets = []
    for tensor in tensors:
        name_offset = builder.CreateString(tensor["name"])
        shape = tensor["shape"]
        shape_count = len(shape)
        TensorStartShapeVector(builder, shape_count)
        for dim in reversed(shape):
            builder.PrependInt32(dim)
        shape_offset = builder.EndVector(shape_count)
        TensorStart(builder)
        TensorAddName(builder, name_offset)
        TensorAddShape(builder, shape_offset)
        TensorAddBuffer(builder, 0)
        tensor_offsets.append(TensorEnd(builder))

    tensors_count = len(tensor_offsets)
    SubGraphStartTensorsVector(builder, tensors_count)
    for tensor_offset in reversed(tensor_offsets):
        builder.PrependUOffsetTRelative(tensor_offset)
    tensors_offset = builder.EndVector(tensors_count)

    inputs_count = len(inputs)
    SubGraphStartInputsVector(builder, inputs_count)
    for input in reversed(inputs):
        builder.PrependInt32(input)
    inputs_offset = builder.EndVector(inputs_count)

    outputs_count = len(outputs)
    SubGraphStartOutputsVector(builder, outputs_count)
    for output in reversed(outputs):
        builder.PrependInt32(output)
    outputs_offset = builder.EndVector(outputs_count)

    OperatorCodeStart(builder)
    OperatorCodeAddBuiltinCode(builder, BuiltinOperator.ADD)
    code0_offset = OperatorCodeEnd(builder)

    ModelStartOperatorCodesVector(builder, 1)
    builder.PrependUOffsetTRelative(code0_offset)
    codes_offset = builder.EndVector(1)

    op_offsets = []
    for op in ops:
        op_inputs = op["inputs"]
        op_inputs_count = len(op_inputs)
        OperatorStartInputsVector(builder, op_inputs_count)
        for op_input in reversed(op_inputs):
            builder.PrependInt32(op_input)
        op_inputs_offset = builder.EndVector(op_inputs_count)

        op_outputs = op["outputs"]
        op_outputs_count = len(op_outputs)
        OperatorStartOutputsVector(builder, op_outputs_count)
        for op_output in reversed(op_outputs):
            builder.PrependInt32(op_output)
        op_outputs_offset = builder.EndVector(op_outputs_count)

        OperatorStart(builder)
        OperatorAddOpcodeIndex(builder, 0)
        OperatorAddInputs(builder, op_inputs_offset)
        OperatorAddOutputs(builder, op_outputs_offset)
        op_offset = OperatorEnd(builder)
        op_offsets.append(op_offset)

    op_offsets_count = len(op_offsets)
    SubGraphStartOperatorsVector(builder, op_offsets_count)
    for op_offset in reversed(op_offsets):
        builder.PrependUOffsetTRelative(op_offset)
    ops_offset = builder.EndVector(op_offsets_count)

    subgraph_name_offset = builder.CreateString("subgraph_name")
    SubGraphStart(builder)
    SubGraphAddName(builder, subgraph_name_offset)
    SubGraphAddTensors(builder, tensors_offset)
    SubGraphAddInputs(builder, inputs_offset)
    SubGraphAddOutputs(builder, outputs_offset)
    SubGraphAddOperators(builder, ops_offset)
    subgraph_offset = SubGraphEnd(builder)

    ModelStartSubgraphsVector(builder, 1)
    builder.PrependUOffsetTRelative(subgraph_offset)
    subgraphs_offset = builder.EndVector(1)

    description_offset = builder.CreateString("model_description")
    ModelStart(builder)
    ModelAddVersion(builder, TFLITE_SCHEMA_VERSION)
    ModelAddOperatorCodes(builder, codes_offset)
    ModelAddSubgraphs(builder, subgraphs_offset)
    ModelAddDescription(builder, description_offset)
    ModelAddBuffers(builder, buffers_offset)
    model_offset = ModelEnd(builder)
    builder.Finish(model_offset)
    model = builder.Output()

    model_root = Model.Model.GetRootAsModel(model, 0)
    fb_model = ModelT.InitFromObj(model_root)

    return fb_model


def test_one_op_model():
    model_def = {
        "tensors": [
            {"name": "model_input", "shape": [1, 224, 224, 3]}, # 0
            {"name": "model_output", "shape": [1, 1001]},       # 1
            {"name": "weights", "shape": [1, 112, 112, 16]},    # 2
        ],
        "inputs": [0],
        "outputs": [1],
        "ops": [
            {"inputs": [0, 2], "outputs": [1]},
        ],
    }
    model = flatbuffer_model_from_def(model_def)
    expected_stats = [(1 * 224 * 224 * 3) + (1 * 1001)]
    actual_stats = model_memory_stats(model)
    for i, actual in enumerate(actual_stats):
        actual_memory = actual[1]
        expected_memory = expected_stats[i]
        assert actual_memory == expected_memory

def test_two_op_model():
    model_def = {
        "tensors": [
            {"name": "model_input", "shape": [1, 224, 224, 3]},   # 0
            {"name": "model_output", "shape": [1, 1001]},         # 1
            {"name": "weights", "shape": [1, 112, 112, 16]},      # 2
            {"name": "intermediate", "shape": [1, 56, 56, 64]},   # 3
        ],
        "inputs": [0],
        "outputs": [1],
        "ops": [
            {"inputs": [0, 2], "outputs": [3]},
            {"inputs": [3], "outputs": [1]},
        ],
    }
    model = flatbuffer_model_from_def(model_def)
    expected_stats = [
        (1 * 224 * 224 * 3) + (1 * 56 * 56 * 64),
        (1 * 56 * 56 * 64) + (1 * 1001),
    ]
    actual_stats = model_memory_stats(model)
    for i, actual in enumerate(actual_stats):
        actual_memory = actual[1]
        expected_memory = expected_stats[i]
        assert actual_memory == expected_memory


def test_alexnet():
    model_def = {
        "tensors": [
            {"name": "model_input", "shape": [1, 224, 224, 3]},   # 0
            {"name": "conv1", "shape": [1, 54, 54, 96]},          # 1
            {"name": "pool1", "shape": [1, 26, 26, 96]},          # 2
            {"name": "conv2", "shape": [1, 26, 26, 256]},         # 3
            {"name": "pool2", "shape": [1, 12, 12, 256]},         # 4
            {"name": "conv3", "shape": [1, 12, 12, 384]},         # 5
            {"name": "conv4", "shape": [1, 12, 12, 384]},         # 6
            {"name": "conv5", "shape": [1, 12, 12, 256]},         # 7
            {"name": "pool5", "shape": [1, 5, 5, 256]},           # 8
            {"name": "fc1", "shape": [1, 4096]},                  # 9
            {"name": "fc2", "shape": [1, 4096]},                  # 10
            {"name": "fc3", "shape": [1, 1000]},                  # 11
        ],
        "inputs": [0],
        "outputs": [11],
        "ops": [
            {"inputs": [0], "outputs": [1]},
            {"inputs": [1], "outputs": [2]},
            {"inputs": [2], "outputs": [3]},
            {"inputs": [3], "outputs": [4]},
            {"inputs": [4], "outputs": [5]},
            {"inputs": [5], "outputs": [6]},
            {"inputs": [6], "outputs": [7]},
            {"inputs": [7], "outputs": [8]},
            {"inputs": [8], "outputs": [9]},
            {"inputs": [9], "outputs": [10]},
            {"inputs": [10], "outputs": [11]},
        ],
    }
    model = flatbuffer_model_from_def(model_def)
    expected_stats = [
        (1 * 224 * 224 * 3) + (1 * 54 * 54 * 96),
        (1 * 54 * 54 * 96) + (1 * 26 * 26 * 96),
        (1 * 26 * 26 * 96) + (1 * 26 * 26 * 256),
        (1 * 26 * 26 * 256) + (1 * 12 * 12 * 256),
        (1 * 12 * 12 * 256) + (1 * 12 * 12 * 384),
        (1 * 12 * 12 * 384) + (1 * 12 * 12 * 384),
        (1 * 12 * 12 * 384) + (1 * 12 * 12 * 256),
        (1 * 12 * 12 * 256) + (1 * 5 * 5 * 256),
        (1 * 5 * 5 * 256) + (1 * 4096),
        (1 * 4096) + (1 * 4096),
        (1* 4096) + (1 * 1000),
    ]
    actual_stats = model_memory_stats(model)
    for i, actual in enumerate(actual_stats):
        actual_memory = actual[1]
        expected_memory = expected_stats[i]
        assert actual_memory == expected_memory


def test_mobilenet_v1():
    model_def = {
        "tensors": [
            {"name": "model_input", "shape": [1, 224, 224, 3]},   # 0
            {"name": "conv1", "shape": [1, 112, 112, 32]},        # 1
            {"name": "dwconv1", "shape": [1, 112, 112, 32]},      # 2
            {"name": "conv2", "shape": [1, 112, 112, 64]},        # 3
            {"name": "dwconv2", "shape": [1, 56, 56, 64]},        # 4
            {"name": "conv3", "shape": [1, 56, 56, 128]},         # 5
            {"name": "dwconv3", "shape": [1, 56, 56, 128]},       # 6
            {"name": "conv4", "shape": [1, 56, 56, 128]},         # 7
            {"name": "dwconv4", "shape": [1, 28, 28, 128]},       # 8
            {"name": "conv5", "shape": [1, 28, 28, 256]},         # 9
            {"name": "dwconv5", "shape": [1, 28, 28, 256]},       # 10
            {"name": "conv6", "shape": [1, 28, 28, 256]},         # 11
            {"name": "dwconv6", "shape": [1, 14, 14, 256]},       # 12
            {"name": "conv7", "shape": [1, 14, 14, 512]},         # 13
            {"name": "dwconv7", "shape": [1, 14, 14, 512]},       # 14
            {"name": "conv8", "shape": [1, 14, 14, 512]},         # 15
            {"name": "dwconv8", "shape": [1, 14, 14, 512]},       # 16
            {"name": "conv9", "shape": [1, 14, 14, 512]},         # 17
            {"name": "dwconv9", "shape": [1, 14, 14, 512]},       # 18
            {"name": "conv10", "shape": [1, 14, 14, 512]},        # 19
            {"name": "dwconv10", "shape": [1, 14, 14, 512]},      # 20
            {"name": "conv11", "shape": [1, 14, 14, 512]},        # 21
            {"name": "dwconv11", "shape": [1, 14, 14, 512]},      # 22
            {"name": "conv12", "shape": [1, 14, 14, 512]},        # 23
            {"name": "dwconv12", "shape": [1, 7, 7, 512]},        # 24
            {"name": "conv13", "shape": [1, 7, 7, 1024]},         # 25
            {"name": "dwconv13", "shape": [1, 7, 7, 1024]},       # 26
            {"name": "conv14", "shape": [1, 7, 7, 1024]},         # 27
            {"name": "pool1", "shape": [1, 1, 1, 1024]},          # 28
            {"name": "conv15", "shape": [1, 1, 1, 1001]},         # 29
            {"name": "reshape1", "shape": [1, 1, 1, 1001]},       # 30
            {"name": "softmax1", "shape": [1, 1, 1, 1001]},       # 31
        ],
        "inputs": [0],
        "outputs": [31],
        "ops": [
            {"inputs": [0], "outputs": [1]},
            {"inputs": [1], "outputs": [2]},
            {"inputs": [2], "outputs": [3]},
            {"inputs": [3], "outputs": [4]},
            {"inputs": [4], "outputs": [5]},
            {"inputs": [5], "outputs": [6]},
            {"inputs": [6], "outputs": [7]},
            {"inputs": [7], "outputs": [8]},
            {"inputs": [8], "outputs": [9]},
            {"inputs": [9], "outputs": [10]},
            {"inputs": [10], "outputs": [11]},
            {"inputs": [11], "outputs": [12]},
            {"inputs": [12], "outputs": [13]},
            {"inputs": [13], "outputs": [14]},
            {"inputs": [14], "outputs": [15]},
            {"inputs": [15], "outputs": [16]},
            {"inputs": [16], "outputs": [17]},
            {"inputs": [17], "outputs": [18]},
            {"inputs": [18], "outputs": [19]},
            {"inputs": [19], "outputs": [20]},
            {"inputs": [20], "outputs": [21]},
            {"inputs": [21], "outputs": [22]},
            {"inputs": [22], "outputs": [23]},
            {"inputs": [23], "outputs": [24]},
            {"inputs": [24], "outputs": [25]},
            {"inputs": [25], "outputs": [26]},
            {"inputs": [26], "outputs": [27]},
            {"inputs": [27], "outputs": [28]},
            {"inputs": [28], "outputs": [29]},
            {"inputs": [29], "outputs": [30]},
            {"inputs": [30], "outputs": [31]},
        ],
    }
    model = flatbuffer_model_from_def(model_def)
    expected_stats = [
        (0, (1 * 224 * 224 * 3) + (1 * 112 * 112 * 32)),
        (1, (1 * 112 * 112 * 32) + (1 * 112 * 112 * 32)),
        (10, (1 * 28 * 28 * 256) + (1 * 28 * 28 * 256)),
        (20, (1 * 14 * 14 * 512) + (1 * 14 * 14 * 512)),
        (30, (1 * 1 * 1 * 1001) + (1 * 1 * 1 * 1001)),
    ]
    actual_stats = model_memory_stats(model)
    for i, expected_memory in expected_stats:
        actual_memory = actual_stats[i][1]
        assert actual_memory == expected_memory
