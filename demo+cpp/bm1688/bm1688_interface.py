



import os
import sophon.sail as sail
import torch
import argparse
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class Data:
    name:str = None
    data:np.array = None
    dtype:np.dtype = None
    qzero_point:int = 0
    qscale:int = 1

type_sail_2_c_dict = {
    sail.Dtype.BM_FLOAT32:"f32",
    sail.Dtype.BM_INT8:"i8",
    sail.Dtype.BM_UINT8:"u8",
    sail.Dtype.BM_INT32:"i32",
    sail.Dtype.BM_UINT32:"u32",
    sail.Dtype.BM_FLOAT16:"f16",
    # sail.Dtype.BM_BFLOAT16:np.dtype('bfloat16'),
    sail.Dtype.BM_INT16:"i16",
    sail.Dtype.BM_UINT16:"u16"
}
type_c_2_np_dict = {
    "f32":np.float32,
    "i8":np.int8,
    "u8":np.uint8,
    "i32":np.int32,
    "u32":np.uint32,
    "f16":np.float16,
    "i16":np.int16,
    "u16":np.uint16
}

class Bm1688Net:
    def __init__(self,name,model):
        self.name = name
        self.model = model
        self.handle = model.get_handle()
        self.inputs = []
        self.outputs = []
        engine = self.model
        for input_name in engine.get_input_names(self.name):
            shape = engine.get_input_shape(self.name,input_name)
            input_type = engine.get_input_dtype(self.name,input_name)
            self.inputs.append(Data(name=input_name,data=np.zeros(shape),dtype=
                    type_sail_2_c_dict[input_type]))

        for output_name in engine.get_output_names(self.name):
            shape = engine.get_output_shape(self.name,output_name)
            output_type = engine.get_output_dtype(self.name,output_name)
            self.outputs.append(Data(name=output_name,data=np.zeros(shape),dtype=
                    type_sail_2_c_dict[output_type]))
        return

    def forward_dynamic(self,input_shapes):
        feed_inputs = {}
        for i in self.inputs:
            feed_inputs[i.name] = np.ascontiguousarray(i.data).astype(type_c_2_np_dict[i.dtype])
        outputs_data = self.model.process(graph_name=self.name,input_tensors=feed_inputs)
        shapes = []
        for o in self.outputs:
            o.data = outputs_data[o.name]
            shapes.append(o.data.shape)

        return shapes
    

class Bm1688Model:
    def __init__(self,model_path,device=0) -> None:
        self.io_mode = sail.IOMode.SYSIO
        self.device = device
        self.engine=sail.Engine(model_path,self.device,self.io_mode)
        self.net = {}
        self.networks = []
        engine = self.engine
        for graph_name in engine.get_graph_names():
            self.net[graph_name] = Bm1688Net(graph_name,engine)
            self.networks.append(graph_name)


    def Net(self,name):
        return self.net[name]

    