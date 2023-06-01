# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ConcatenationOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsConcatenationOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ConcatenationOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ConcatenationOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # ConcatenationOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ConcatenationOptions
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ConcatenationOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def ConcatenationOptionsStart(builder): builder.StartObject(2)
def ConcatenationOptionsAddAxis(builder, axis): builder.PrependInt32Slot(0, axis, 0)
def ConcatenationOptionsAddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(1, fusedActivationFunction, 0)
def ConcatenationOptionsEnd(builder): return builder.EndObject()


class ConcatenationOptionsT(object):

    # ConcatenationOptionsT
    def __init__(self):
        self.axis = 0  # type: int
        self.fusedActivationFunction = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        concatenationOptions = ConcatenationOptions()
        concatenationOptions.Init(buf, pos)
        return cls.InitFromObj(concatenationOptions)

    @classmethod
    def InitFromObj(cls, concatenationOptions):
        x = ConcatenationOptionsT()
        x._UnPack(concatenationOptions)
        return x

    # ConcatenationOptionsT
    def _UnPack(self, concatenationOptions):
        if concatenationOptions is None:
            return
        self.axis = concatenationOptions.Axis()
        self.fusedActivationFunction = concatenationOptions.FusedActivationFunction()

    # ConcatenationOptionsT
    def Pack(self, builder):
        ConcatenationOptionsStart(builder)
        ConcatenationOptionsAddAxis(builder, self.axis)
        ConcatenationOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        concatenationOptions = ConcatenationOptionsEnd(builder)
        return concatenationOptions
