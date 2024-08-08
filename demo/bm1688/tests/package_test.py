
from bm1688 import bm1688_pyruntime
import numpy as np
from utils import get_input,get_output

## 如果要做测试，请将其移动到demo.py同目录下
# bm1688_pyruntime.Model("./mlir/MobileVLM_V2-1.7B-F16.bmodel")
# model=bm1688_pyruntime.Model("./mlir/embedding.bmodel")
# inputs=np.load("./test_case_data/embedding_input.npz")
    

# net = model.Net(model.networks[0])
# print(net)

# shape = get_input(inputs,net)
# shape = net.forward_dynamic(shape)
# outputs = get_output(shape,net)

# for key in outputs:
#     print(outputs[key])