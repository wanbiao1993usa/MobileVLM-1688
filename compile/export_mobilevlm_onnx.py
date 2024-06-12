#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

#export PYTHONPATH=/workspace/chatglm2-6b:$PYTHONPATH
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

## 需要加载模型
model_path = "/home/wanbiao/workspace/MobileVLM/MobileVLM_V2-1.7B"
folder = "./mobilevlm-tmp"

origin_model = MobileLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,device_map='auto',torch_dtype= torch.float)
origin_model.eval()

##模型结构
## 1. Origin_model:MobileLlamaForCausalLM
##          model:MobileLlamaModel
##          lm_head:Linear
## 2. model:MobileLlamModel
##          继承：MobileVLMMetaModel
##          继承：LlamaModel
## 3. MobileVLMMetaModel
##          vision_tower:CLIPVisionTower
##          mm_projector:LDPNetV2Projector
## 4. LlamaMode
##          embed_tokens:Embedding
##          layers:LlamaDecoderLayer * 24
##          norm:LlamaRMSNorm

## 将LlamaMode中的Embedding转成ONNX
embedding_model = origin_model.model.embed_tokens
embedding_model_path = f'./tmp/embedding.onnx'
torch.onnx.export(embedding_model, 
                  (torch.tensor([0, 1, 2, 3])),##[1,max_sequence]==[1,2048]
                      embedding_model_path,
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],##[1,max_sequence,embeding]=[1,2048,2048]
                      dynamic_axes={"input_ids": {
                          0: "length"
                      }},
                      do_constant_folding=True,
                      opset_version=15)
##测试转换是否正确
# embedding_model_onnx = onnxruntime.InferenceSession(embedding_model_path)
# embedding_model_onnx.run(None,{'input_ids':[1,2,3]})

## 将LlamaModel中的layers转成ONNX
layers = origin_model.model.layers
max_sequence = origin_model.config.max_sequence_length ## 2048
dim = origin_model.config.hidden_size
torch.cuda.empty_cache()
for index,layer in enumerate(layers):
    print(f"dump {index} layer")
    # input
    hidden_states = torch.randn((1,max_sequence,dim))
    position_ids = torch.tensor([range(max_sequence)], dtype=torch.long)
    attention_mask = torch.ones((1, 1, max_sequence, max_sequence),dtype=torch.bool).tril(diagonal=0)

    torch.onnx.export(layer,
                    (hidden_states, attention_mask,position_ids),
                      f'./tmp/layer_{index}.onnx',
                      verbose=False,
                      input_names=['input_states','attention_mask', 'position_ids'],
                      output_names=['hidden_states'],
                      do_constant_folding=True,
                      opset_version=15)
    torch.cuda.empty_cache()

## 将LlamaModel中的Norm转成ONNX
norm = origin_model.model.norm
hidden_states = torch.randn((1,max_sequence,dim))
torch.onnx.export(norm,
                    (hidden_states),
                      f'./tmp/norm.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['token'],
                      do_constant_folding=True,
                      opset_version=15)
torch.cuda.empty_cache()

## 將MobileVLMMetaModel中的vision_tower转成ONNX
vision_tower = origin_model.model.vision_tower
dummy_image = torch.zeros(1, 3, 336, 336)
torch.onnx.export(vision_tower,
                    (dummy_image),
                      f'./tmp/vision_tower.onnx',
                      verbose=False,
                      input_names=['image_tensor'],
                      output_names=['image_features'],
                      do_constant_folding=True,
                      opset_version=15)
torch.cuda.empty_cache()

## 将MobileVLMMetaModel中的mm_projector转成ONNX
mm_projector = origin_model.model.mm_projector
hidden_feature = torch.zeros(1,576,1024) ## output [1,144,2048]
torch.onnx.export(mm_projector,
                    (hidden_feature),
                      f'./tmp/mm_projector.onnx',
                      verbose=False,
                      input_names=['hidden_feature'],
                      output_names=['image_features'],
                      do_constant_folding=True,
                      opset_version=15)
torch.cuda.empty_cache()


## 将MobileLlamaForCausalLM的lm_head转成ONNX
lm_head = origin_model.lm_head
hidden_feature = torch.randn(1, dim)
torch.onnx.export(lm_head, (hidden_feature),
                    f'./tmp/lm_head.onnx',
                    verbose=False,
                    input_names=['hidden_states'],
                    output_names=['token_ids'],
                    do_constant_folding=True,
                    opset_version=15)

print("模型转换完成🚀🚀🚀")