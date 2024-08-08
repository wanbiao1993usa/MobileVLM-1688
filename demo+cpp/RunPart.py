
from bm1688 import bm1688_pyruntime
import numpy as np
from utils import get_input,get_output
from utils import  get_input_ids, load_tokenizer
from mobilevlm.constants import DEFAULT_IMAGE_TOKEN
import argparse
from mobilevlm.conversation import conv_templates, SeparatorStyle


class RunPart:
    def __init__(self,args):
        self.input_file = args.input
        self.output_file = args.output
        self.model_path = args.model
        self.model = bm1688_pyruntime.Model(self.model_path)
        
    def process(self):
        net = self.model.Net(self.model.networks[0])
        inputs=np.load(self.input_file)
        shape = get_input(inputs,net)
        shape = net.forward_dynamic(shape)
        outputs = get_output(shape,net)
        np.savez(self.output_file,**outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行bmodel的一部分")
    parser.add_argument("--input",type=str,help="输入的npz文件")
    parser.add_argument("--model",type=str,help="输入的模型路径")
    parser.add_argument("--output",type=str,help="保存路径")
    args = parser.parse_args()
    print(f"args is:{args}")

    RunPart(args).process()


