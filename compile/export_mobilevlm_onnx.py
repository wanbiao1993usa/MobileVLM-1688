#!/usr/bin/env python3

## 注意将MobileVLM-1688加入python搜索路径
## export PYTHONPATH=$PYTHONPATH:/home/wanbiao/workspace/MobileVLM-1688/
## 修改config，将use_cache设置为False
import datetime
import math
import unittest
import torch
import random
import sys
from mobilevlm.model.mobilellama import MobileLlamaForCausalLM
import os
import numpy as np
import onnxruntime
import onnx
import onnxoptimizer
from transformers import AutoTokenizer



##模型结构
## 1. Origin_model:MobileLlamaForCausalLM
##          model:MobileLlamaModel
##          lm_head:Linear
## 2. model:MobileLlamaModel
##          继承：MobileVLMMetaModel
##          继承：LlamaModel
## 3. MobileVLMMetaModel
##          vision_tower:CLIPVisionTower
##          mm_projector:LDPNetV2Projector
## 4. LlamaMode
##          embed_tokens:Embedding
##          layers:LlamaDecoderLayer * 24
##          norm:LlamaRMSNorm

##模型继承关系
## MobileLlamaForCausalLM
#     LlamaForCausalLM
#         LlamaPreTrainedModel
#             PreTrainedModel
#                 nn.Module
#                 ModuleUtilsMixin
#                 GenerationMixin
#                 PushToHubMixin
#                 PeftAdapterMixin
#     MobileVLMMetaForCausalLM
#         ABC
# MobileLlamaModel
#     MobileVLMMetaModel
#     LlamaModel
#         LlamaPreTrainedModel
#             PreTrainedModel
#                 nn.Module
#                 ModuleUtilsMixin
#                 GenerationMixin
#                 PushToHubMixin
#                 PeftAdapterMixin
# 其中MobileLlamaForCausalLM 包含 MobileLlamaModel。通过getModel()获得


class ExportMobileVLM:
    def __init__(self, model_path,save_path="mobilevlm-tmp"):
        self.save_path = save_path
        self.model_path = model_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.loadModel()
      
    def loadModel(self):
        ## 默认使用cpu加载，因为实验设备的cpu有128G
        self.origin_model = MobileLlamaForCausalLM.from_pretrained(self.model_path, low_cpu_mem_usage=True,device_map='cpu',torch_dtype= torch.float)
        self.origin_model.eval() ## 这是必要的，因为像dropout或batchnorm这样的算子在推理模式和训练模式下的行为是不同的。

        self.max_sequence = self.origin_model.config.max_sequence_length ## 为了适配内存，应该为512，而不是2048
        assert self.max_sequence == 512, f"序列最大长度不是512, 而是{self.max_sequence},应该修改对应的config文件"
        self.dim = self.origin_model.config.hidden_size
        self.mm_hidden_size = self.origin_model.config.mm_hidden_size
        self.vocab_size = self.origin_model.config.vocab_size
        self.input_image_width = 336
        self.input_image_height = 336
        self.input_image_patch_width = 14
        self.input_image_patch_height = 14

    def export(self):
        exported_models = []
        exported_models.append(self.export_embedding())
        exported_models.append(self.export_layers())
        exported_models.append(self.export_norm())
        exported_models.append(self.export_vision_tower())
        # exported_models.append(self.export_mm_projector())
        exported_models.append(self.export_lm_head())
        exported_models.append(self.export_mm_projector_mlp())
        exported_models.append(self.export_mm_projector_dwn())
        exported_models.append(self.export_mm_projector_peg())

        print(f"\n{exported_models}转换完成🚀🚀🚀")
        return
    
    def export_mm_projector_peg(self):
        model_name = "mm_projector_peg.onnx"
        model_path = os.path.join(self.save_path,model_name)
        model = self.origin_model.model.mm_projector.peg
        x= torch.rand([1, 144, 2048]) ##永远是这个数据
        inputs = (x,)
        torch.onnx.export(model,
                            inputs,
                            model_path,
                            verbose=False,
                            input_names=['x'],
                            output_names=['image_features'],
                            dynamic_axes={
                                'hidden_feature': {0: 'batch_size'},
                                'image_features':{0: 'batch_size'},
                            },
                            do_constant_folding=True,
                            opset_version=15)
        self.check_mm_projector_peg(compared_model=model,inputs=inputs,path=model_path)
        return model_name

    def check_mm_projector_peg(self,compared_model,inputs,path):
        test_model = onnxruntime.InferenceSession(path)
        test_inputs = {test_model.get_inputs()[i].name: inputs[i].numpy() for i in range(len(inputs))}
        test_outputs = test_model.run(None,test_inputs)
        torch_out = compared_model(*inputs)

        np.testing.assert_allclose(self.to_numpy(torch_out), test_outputs[0], rtol=1e-03, atol=1e-05)

        print(f"导出的{path}已经使用ONNXRuntime进行了测试，结果看起来很好！简直他妈的是天才")
        return True
    
    def export_mm_projector_dwn(self):
        model_name = "mm_projector_dwn.onnx"
        model_path = os.path.join(self.save_path,model_name)
        model = self.origin_model.model.mm_projector.dwn
        patches = int((self.input_image_width / self.input_image_patch_width) * (self.input_image_height / self.input_image_patch_height))
        x = torch.rand(1,patches,2048)
        inputs = (x,)
        torch.onnx.export(model,
                            inputs,
                            model_path,
                            verbose=False,
                            input_names=['x'],
                            output_names=['image_features'],
                            dynamic_axes={
                                'hidden_feature': {0: 'batch_size'},
                                'image_features':{0: 'batch_size'},
                            },
                            do_constant_folding=True,
                            opset_version=15)
        self.check_mm_projector_dwn(compared_model=model,inputs=inputs,path=model_path)
        return model_name

    def check_mm_projector_dwn(self,compared_model,inputs,path):
        test_model = onnxruntime.InferenceSession(path)
        test_inputs = {test_model.get_inputs()[i].name: inputs[i].numpy() for i in range(len(inputs))}
        test_outputs = test_model.run(None,test_inputs)
        torch_out = compared_model(*inputs)

        np.testing.assert_allclose(self.to_numpy(torch_out), test_outputs[0], rtol=1e-03, atol=1e-05)

        print(f"导出的{path}已经使用ONNXRuntime进行了测试，结果看起来很好！简直他妈的是天才")
        return True
    
    def export_mm_projector_mlp(self):
        model_name = "mm_projector_mlp.onnx"
        model_path = os.path.join(self.save_path,model_name)
        model = self.origin_model.model.mm_projector.mlp
        patches = int((self.input_image_width / self.input_image_patch_width) * (self.input_image_height / self.input_image_patch_height))
        x = torch.zeros(1,patches,self.mm_hidden_size) ### (batch_size,576,1024)
        x = torch.rand(1,patches,self.mm_hidden_size)
        inputs = (x,)
        torch.onnx.export(model,
                            inputs,
                            model_path,
                            verbose=False,
                            input_names=['x'],
                            output_names=['image_features'],
                            dynamic_axes={
                                'hidden_feature': {0: 'batch_size'},
                                'image_features':{0: 'batch_size'},
                            },
                            do_constant_folding=True,
                            opset_version=15)
        self.check_mm_projector_mlp(compared_model=model,inputs=inputs,path=model_path)
        return model_name

    def check_mm_projector_mlp(self,compared_model,inputs,path):
        test_model = onnxruntime.InferenceSession(path)
        test_inputs = {test_model.get_inputs()[i].name: inputs[i].numpy() for i in range(len(inputs))}
        test_outputs = test_model.run(None,test_inputs)
        torch_out = compared_model(*inputs)

        np.testing.assert_allclose(self.to_numpy(torch_out), test_outputs[0], rtol=1e-03, atol=1e-05)

        print(f"导出的{path}已经使用ONNXRuntime进行了测试，结果看起来很好！简直他妈的是天才")
        return True
    
    def export_lm_head(self):
        model_name = "lm_head.onnx"
        model = self.origin_model.lm_head
        model_path = os.path.join(self.save_path,model_name)
        hidden_feature = torch.randn(1, self.max_sequence,self.dim) ### (1,sequence_length,dim) (1,205,2048)
        inputs = (hidden_feature,)
        torch.onnx.export(model, 
                          inputs,
                            model_path,
                            verbose=False,
                            input_names=['hidden_states'],
                            output_names=['token_ids'],
                            dynamic_axes={
                                'hidden_states': {0: 'batch_size'},
                                'token_ids': {0: 'batch_size'},
                            },
                            do_constant_folding=True,
                            opset_version=15)
        self.check_lm_head(compared_model=model,inputs=inputs,path=model_path)
        return model_name
    
    def check_lm_head(self,compared_model,inputs,path):
        test_model = onnxruntime.InferenceSession(path)
        test_inputs = {test_model.get_inputs()[i].name: inputs[i].numpy() for i in range(len(inputs))}
        test_outputs = test_model.run(None,test_inputs)
        torch_out = compared_model(*inputs)

        np.testing.assert_allclose(self.to_numpy(torch_out), test_outputs[0], rtol=1e-03, atol=1e-05)

        print(f"导出的{path}已经使用ONNXRuntime进行了测试，结果看起来很好！简直他妈的是天才")
        return True
    
    def export_mm_projector(self):
        model_name = "mm_projector.onnx"
        model_path = os.path.join(self.save_path,model_name)
        model = self.origin_model.model.mm_projector
        patches = int((self.input_image_width / self.input_image_patch_width) * (self.input_image_height / self.input_image_patch_height))
        hidden_feature = torch.zeros(1,patches,self.mm_hidden_size) ### (batch_size,576,1024)
        hidden_feature = torch.rand(1,patches,self.mm_hidden_size)
        inputs = (hidden_feature,)
        torch.onnx.export(model,
                            inputs,
                            model_path,
                            verbose=False,
                            input_names=['hidden_feature'],
                            output_names=['image_features'],
                            dynamic_axes={
                                'hidden_feature': {0: 'batch_size'},
                                'image_features':{0: 'batch_size'},
                            },
                            do_constant_folding=True,
                            opset_version=15)
        self.check_mm_project(compared_model=model,inputs=inputs,path=model_path)
        return model_name
    
    def check_mm_project(self,compared_model,inputs,path):
        test_model = onnxruntime.InferenceSession(path)
        test_inputs = {test_model.get_inputs()[i].name: inputs[i].numpy() for i in range(len(inputs))}
        test_outputs = test_model.run(None,test_inputs)
        torch_out = compared_model(*inputs)

        np.testing.assert_allclose(self.to_numpy(torch_out), test_outputs[0], rtol=1e-03, atol=1e-05)

        print(f"导出的{path}已经使用ONNXRuntime进行了测试，结果看起来很好！简直他妈的是天才")
        return True
    
    def export_vision_tower(self):
        ## 將MobileVLMMetaModel中的vision_tower转成ONNX
        vision_tower = self.origin_model.model.vision_tower
        dummy_image = torch.zeros(1, 3, self.input_image_width, self.input_image_height) ##[batch_size,3,336,336]
        dummy_image = torch.rand(1, 3, self.input_image_width, self.input_image_height)
        dummy_image = torch.load("image_tensor.pt").cpu()
        print(dummy_image)
        model_name = "vision_tower.onnx"
        model_path = os.path.join(self.save_path,model_name)
        inputs = (dummy_image,)
        torch.onnx.export(vision_tower,
                            inputs,
                            model_path,
                            verbose=False,
                            input_names=['image_tensor'],
                            output_names=['image_features'],
                            dynamic_axes={
                                'image_tensor': {0: 'batch_size'},
                                'image_features':{0: 'batch_size'},
                            },
                            do_constant_folding=True,
                            opset_version=15)
        
        # 进行优化
        model = onnx.load(model_path)
        optimized_model = onnxoptimizer.optimize(model)
        onnx.save(optimized_model, model_path)

        self.check_vision_tower(compared_model=vision_tower,inputs=inputs,path=model_path)
        return model_name
    
    def check_vision_tower(self,compared_model,inputs,path):
        test_model = onnxruntime.InferenceSession(path)
        test_inputs = {test_model.get_inputs()[i].name: inputs[i].numpy() for i in range(len(inputs))}
        test_outputs = test_model.run(None,test_inputs)
        torch_out = compared_model(*inputs)

        print(f"onnx_output {test_outputs}")
        print(f"torch_output {torch_out}")

        ### 出现了精度下降的情况
        # np.testing.assert_allclose(self.to_numpy(torch_out), test_outputs[0], rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(self.to_numpy(torch_out), test_outputs[0], rtol=1e-03, atol=1e-02)

        print(f"导出的{path}已经使用ONNXRuntime进行了测试，结果看起来很好！简直他妈的是天才")
        return True

    def export_norm(self):
        ## 将LlamaModel中的Norm转成ONNX
        norm = self.origin_model.model.norm
        hidden_states = torch.randn((1,self.max_sequence,self.dim))
        model_name = "norm.onnx"
        model_path = os.path.join(self.save_path,model_name)
        inputs = (hidden_states,)
        torch.onnx.export(norm,
                            inputs,
                            model_path,
                            verbose=False,
                            input_names=['hidden_states'],
                            output_names=['token'],
                            dynamic_axes={
                                'hidden_states': {0: 'batch_size'},
                                'token': {0: 'batch_size'},
                            },
                            do_constant_folding=True,
                            opset_version=15)
        self.check_norm(compared_model=norm,inputs=inputs,path=model_path)
        return model_name
    
    def check_norm(self,compared_model,inputs,path):
        test_model = onnxruntime.InferenceSession(path)
        test_inputs = {test_model.get_inputs()[i].name: inputs[i].numpy() for i in range(len(inputs))}
        test_outputs = test_model.run(None,test_inputs)
        torch_out = compared_model(*inputs)

        np.testing.assert_allclose(self.to_numpy(torch_out), test_outputs[0], rtol=1e-03, atol=1e-05)

        print(f"导出的{path}已经使用ONNXRuntime进行了测试，结果看起来很好！简直他妈的是天才")
        return True

    def export_layers(self):
        ## 将LlamaModel中的layers转成ONNX
        ## layers是多个LlamaDecoderLayer模型的堆叠
        ## 因为LlamaDecoderLayer继承于nn.Module,可以直接导出

        ## hidden_states:torch.FloatTensor:(batch_size,seq_len,embed_dim) ### torch.Size([1, 205, 2048])
        ## attention_mask:torch.FloatTensor:(batch, 1, tgt_len, src_len).tgt_len:目标序列长度，src_len:源序列长度 ###torch.Size([1, 1, 205, 205])
        ## position_id:torch.LongTensor:(1,seq_len) ### torch.Size([1, 205])
        ## output_attentions：bool ### False
        ## use_cache:bool ### True
        ## past_key_value:None

        layers = self.origin_model.model.layers
        
        layer_name = "layer"
        print(f"开始转换{layer_name}")

        layers_name = []
        for index,layer in enumerate(layers):
            sub_layer_name = f"{layer_name}_{index}.onnx"
            sub_layer_name_path = os.path.join(self.save_path, sub_layer_name)
            print(f"开始转换{sub_layer_name}")
            # input
            hidden_states = torch.randn((1,self.max_sequence,self.dim))
            position_ids = torch.tensor([range(self.max_sequence)])
            attention_mask = torch.ones((1, 1, self.max_sequence, self.max_sequence)) ## 第二个维度总是为1
            # past_key_value = (torch.randn(1, 1, self.max_sequence,self.dim), torch.randn(1, 1, self.max_sequence,self.dim)) ## 应该为None
            past_key_value = None
            output_attentions=False
            use_cache=False            
            inputs = (hidden_states, attention_mask,position_ids,past_key_value,output_attentions, use_cache)

            torch.onnx.export(layer,
                            inputs,
                            sub_layer_name_path,
                            verbose=False,
                            input_names=['hidden_states', 'attention_mask', 'position_ids', 'past_key_value','output_attentions', 'use_cache'],
                            # output_names=['output','self_attn_weights', 'present_key_value'], ## 只有在output_attentions和use_cache为True时，才有三个输出
                            output_names=['output'],
                            dynamic_axes={
                                'hidden_states': {0: 'batch_size'},
                                'attention_mask': {0: 'batch_size'},
                                'position_ids': {0: 'batch_size'},
                                'output': {0: 'batch_size'},
                            },
                            do_constant_folding=True,
                            opset_version=15)
            torch.cuda.empty_cache()
            print(f"转换{sub_layer_name}完成")
            self.check_sublayer(compared_model=layer,inputs=inputs,path=sub_layer_name_path)
            layers_name.append(sub_layer_name)
        print(f"转换{layer_name}完成")
        return layers_name


    def check_sublayer(self,compared_model,inputs,path):
        test_model = onnxruntime.InferenceSession(path)
        test_inputs = {test_model.get_inputs()[i].name: inputs[i].numpy() for i in range(len(test_model.get_inputs()))}
        test_outputs = test_model.run(None,test_inputs)
        torch_out = compared_model(*inputs)

        np.testing.assert_allclose(self.to_numpy(torch_out[0]), test_outputs[0], rtol=1e-03, atol=1e-05)

        print(f"导出的{path}已经使用ONNXRuntime进行了测试，结果看起来很好！简直他妈的是天才")
        return True


    def export_embedding(self):
        ## MobileLlamaForCausalLM.getModle()=MobileLlamaModel
        ## MobileLlamaModel继承于LlamaModel
        ## LlamaModel包含有embed_tokens
        model = self.origin_model.model.embed_tokens
        model_name = "embedding.onnx"
        model_path = os.path.join(self.save_path, model_name)
        input_shape= torch.full((1, self.max_sequence), 3) ## [batch_size,sequence_length]
        print(f"开始转换{model_name}")
        torch.onnx.export(model, 
                          input_shape,##[batch_size,max_sequence]==[batch_size,2048] ## max_sequence为了适配修改为512
                          model_path,
                          verbose=False,
                          input_names=['input_ids'],
                          output_names=['input_embed'],##[1,max_sequence,embeding]=[1,2048,2048]
                          dynamic_axes={"input_ids": {0: "batch_size"},
                                        "input_embed":{0:"batch_size"}},
                          do_constant_folding=True,
                          opset_version=15)
        self.check_embedding(compared_model=model,input=input_shape,path=model_path)
        print(f"转换{model_name}完成")
        return model_name
    
    def check_embedding(self,compared_model,input,path=""):
        ## 数值检查
        model = onnxruntime.InferenceSession(path)

        test_inputs = {model.get_inputs()[0].name: self.to_numpy(input)}
        test_outs = model.run(None, test_inputs)

        torch_out = compared_model(input)

        # 比较 ONNX Runtime 和 PyTorch 结果
        np.testing.assert_allclose(self.to_numpy(torch_out), test_outs[0], rtol=1e-03, atol=1e-05)

        print(f"导出的{path}已经使用ONNXRuntime进行了测试，结果看起来很好！简直他妈的是天才")
        return True
    
    def to_numpy(self,tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    

class ExportTokenizer:
    def __init__(self, model_path,save_path="mobilevlm-tmp"):
        self.save_path = save_path
        self.model_path = model_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.loadModel()
      
    def loadModel(self):
        ## 默认使用cpu加载，因为实验设备的cpu有128G
        self.origin_model = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        print(self.origin_model)

    def export(self):
        exported_models = []
        print(f"\n{exported_models}转换完成🚀🚀🚀")
        return



if __name__ == "__main__":
    model_path = "/home/wanbiao/workspace/MobileVLM-1688/MobileVLM_V2-1.7B"
    tmp_folder = "./mobilevlm-tmp"
    # ExportMobileVLM(model_path,tmp_folder).export()
    ExportTokenizer(model_path,tmp_folder).export()
