import os
import numpy as np
from PIL import Image
from utils import get_image_tensor, get_input, get_input_ids, get_output, load_tokenizer,  model_import

from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mobilevlm.utils import tokenizer_image_token, KeywordsStoppingCriteria
import torch
from typing import  Optional
import subprocess

from mobilevlm.conversation import conv_templates, SeparatorStyle

class BmBaseModel:
    def __init__(self,model):
        self.mode_path = model

    def run(self,inputs):
        if self.__class__.__name__ == "VisionTower":
            a = np.load("./test_case_data/vision_tower_output.npz")
            return a
        self.correct_type(inputs)
        self.prepare_input(inputs)
        # result = subprocess.run(['python3','demo+cpp/RunPart.py', 
        #                          f"--input=input.npz",
        #                          f"--model={self.mode_path}",
        #                          f"--output=tmp.npz"], capture_output=True, text=True)
        result = subprocess.run(['./cpp-demo/mobilevlm.soc',
                                 f"--input=input.npz",
                                 f"--model={self.mode_path}",
                                 f"--output=tmp.npz"], capture_output=True, text=True)
        print(f"result {result}")
        return self.get_output()
        
    
    def get_output(self):
        result = np.load("tmp.npz")
        return result

    def __call__(self,inputs):
        return self.run(inputs=inputs)
    
    def prepare_input(self,inputs):
        if hasattr(self, 'get_correct_input') and callable(getattr(self, 'get_correct_input')):
            self.get_correct_input(inputs)
        else:
            pass
        for k in inputs:
            if inputs[k].dtype == np.float16:
                inputs[k] = inputs[k].astype(np.float32)
        np.savez("input.npz",**inputs)

    def correct_type(self,inputs):
        keys=list(inputs.keys())
        for i in range(len(self.dtype)):
            inputs[keys[i]] = inputs[keys[i]].astype(self.dtype[i])
            

    
class Embedding(BmBaseModel):
    def __init__(self,model):
        self.dtype = [np.int32]
        model_path = os.path.join(model,"embedding.bmodel")
        super().__init__(model=model_path)
    
    def get_correct_input(self,inputs):
        #数据填充到512
        for key in inputs:
            input = inputs[key]
            padding_width = 512 - input.shape[1]
            inputs[key] = np.pad(input, ((0, 0), (0, padding_width)), 'constant', constant_values=0)
            ## 如果有填充的Id需要，改成0
            inputs[key][inputs[key] < 0] = 0
        return inputs

class VisionTower(BmBaseModel):
    def __init__(self,model):
        self.dtype = [np.float16]
        model_path = os.path.join(model,"vision_tower.bmodel")
        super().__init__(model=model_path)

class MmProjectorMLP(BmBaseModel):
    def __init__(self,model):
        self.dtype = [np.float16]
        model_path = os.path.join(model,"mm_projector_mlp.bmodel")
        super().__init__(model=model_path)

class MmProjectorDWN(BmBaseModel):
    def __init__(self,model):
        self.dtype = [np.float16]
        model_path = os.path.join(model,"mm_projector_dwn.bmodel")
        super().__init__(model=model_path)

class MmProjectorPEG(BmBaseModel):
    def __init__(self,model):
        self.dtype = [np.float16]
        model_path = os.path.join(model,"mm_projector_peg.bmodel")
        super().__init__(model=model_path)

class MmProjector(BmBaseModel):
    def __init__(self,model,save_result=True,test=True):
        self.mlp = MmProjectorMLP(model)
        self.dwn = MmProjectorDWN(model)
        self.peg = MmProjectorPEG(model)

    def __call__(self, inputs) -> dict:
        return self.forward(inputs)
        

    def forward(self,inputs):
        output = self.mlp({"x":list(inputs.values())[0]})
        output = self.dwn({"x":list(output.values())[0]})
        output = self.peg({"x":list(output.values())[0]})
        return output

class Layer(BmBaseModel):
    def __init__(self,model,index=0):
        self.index = index
        self.dtype = [np.float16,np.float16,np.int32]
        model_path = os.path.join(model,f"layer_{index}.bmodel")
        super().__init__(model=model_path)

class Norm(BmBaseModel):
    def __init__(self,model):
        self.dtype = [np.float16]
        model_path = os.path.join(model,"norm.bmodel")
        super().__init__(model=model_path)

class LmHead(BmBaseModel):
    def __init__(self,model):
        self.dtype = [np.float16]
        model_path = os.path.join(model,"lm_head.bmodel")
        super().__init__(model=model_path)


class LlamaModel:
    def __init__(self,model,**args):
        self.layers = []
        self.layer_num = args["layers"]
        for i in range(self.layer_num):
            layer_model = Layer(model,index=i)
            self.layers.append(layer_model)
        self.norm = Norm(model=model)

    
    def __call__(self, inputs_embeds,attention_mask) -> dict:
        return self.forward(inputs_embeds,attention_mask)
        

    def forward(self,inputs_embeds,attention_mask):
        inputs_embeds = torch.from_numpy(inputs_embeds)
        attention_mask = torch.from_numpy(attention_mask)
        batch_size, seq_length, _ = inputs_embeds.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = hidden_states.numpy() if isinstance(hidden_states, torch.Tensor) else hidden_states
            attention_mask = attention_mask.numpy() if isinstance(attention_mask, torch.Tensor) else attention_mask
            position_ids = position_ids.numpy() if isinstance(position_ids, torch.Tensor) else position_ids
            layer_outputs = decoder_layer({
                "hidden_states":hidden_states,
                "attention_mask":attention_mask,
                "position_ids":position_ids
                })
            
            for key in layer_outputs:
                hidden_states = layer_outputs[key]
                break

        hidden_states = self.norm({"hidden_states":hidden_states})

        return hidden_states


    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    def _make_causal_mask(
        self,input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
    ):
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    
    def _expand_mask(self,mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    

class MobileVLM:
    def __init__(self,**args):

        self.pad_token_id = 0
        self.eos_token_id = 2

        self.model_path = args["model_path"]
        self.max_sequence_len = args["max_seq_len"]

        # 模型名
        # 模型
        self.embedding_model = Embedding(self.model_path)
        self.vision_tower_model = VisionTower(self.model_path)
        self.mm_projector_model = MmProjector(self.model_path)
        self.lm_head_model = LmHead(self.model_path,)
        self.model_llama = LlamaModel(model=self.model_path,**args)

    def forward(self,prompt:str,images:list):
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        self.prompt = conv.get_prompt()
        self.stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        # 获取input_id
        self.tokenizer = load_tokenizer("/home/linaro/workspace/MobileVLM-1688/mlir")
        self.input_ids = get_input_ids(tokenizer=self.tokenizer,prompt=self.prompt)
        self.stopping_criteria = KeywordsStoppingCriteria([self.stop_str], self.tokenizer, self.input_ids)
        self.origin_input_ids = self.input_ids

        new_images = []
        for image in images:
            new_images.append(Image.open(image).convert("RGB"))
        self.new_images_tensor = get_image_tensor(new_images)


        ## 跟踪结束
        unfinished_sequences = torch.ones(self.input_ids.shape[0], dtype=torch.long)

        actual_len = self.input_ids.shape[1]
        padding_width = self.max_sequence_len - actual_len
        self.input_ids = np.pad(self.input_ids, ((0, 0), (0, padding_width)), 'constant', constant_values=self.pad_token_id)

        this_peer_finished = False
        ## 自回归调用模型
        while True:
            attention_mask = np.ones((1, actual_len))
            padding_width = self.max_sequence_len - actual_len
            attention_mask = np.pad(attention_mask, ((0, 0), (0, padding_width)), 'constant', constant_values=0)

            attention_mask,input_embedding,new_actual_len = self.prepare_inputs_for_multimodal(
                self.input_ids,attention_mask,self.new_images_tensor,actual_len)
            
            print(f"new_actual_len {new_actual_len}")
            
            outputs = self.once_forward(input_embedding,attention_mask)

            for key in outputs:
                outputs = outputs[key]
                break
            next_token_logits = outputs[:, new_actual_len-1, :]
            next_tokens_scores = torch.from_numpy(next_token_logits)

            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            
            next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)

            self.input_ids[0][actual_len] = next_tokens
            actual_len = actual_len + 1

            eos_token_id_tensor = torch.tensor([self.eos_token_id])
            unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )
            if unfinished_sequences.max() == 0:
                this_peer_finished = True
            
            if actual_len >= self.max_sequence_len:
                this_peer_finished = True
            
            if this_peer_finished:
                break

        input_token_len = self.input_ids.shape[1]-self.origin_input_ids.shape[1]
        
        outputs = self.tokenizer.batch_decode(self.input_ids[:, -input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(self.stop_str):
            outputs = outputs[: -len(self.stop_str)]
        print(f"🚀 MobileVLM_V2-1.7B-F16: {outputs.strip()}\n")

            

    def once_forward(self,inputs_embeds,attention_mask):
        output = self.model_llama(inputs_embeds,attention_mask)
        return self.lm_head_model({"hidden_states":list(output.values())[0]})
        
    def encode_images(self,images):
        output = self.vision_tower_model({"image_tensor":images})
        output = self.mm_projector_model(output)
        return output

    def prepare_inputs_for_multimodal(self,input_ids, attention_mask, images,actual_len):

        ##压缩image
        image_features = self.encode_images(images)
        image_features = list(image_features.values())[0]

        origin_len = actual_len
        new_actual_len = actual_len

        new_input_embeds = []
        end_input_embeds = []

        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_input_ids = torch.from_numpy(cur_input_ids)
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            end_new_input_embeds = []
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                
                output = self.embedding_model({"input_ids":cur_input_ids[:image_token_start].reshape(1, -1).numpy()})
                for k in output:
                    output_value = output[k].squeeze()
                    output_value = output_value[:image_token_start]
                    cur_new_input_embeds.append(output_value)

                cur_new_input_embeds.append(cur_image_features)

                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start+1:]
                actual_len = actual_len - (image_token_start+1)
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                
                start_ids = cur_input_ids[:actual_len]
                end_ids = cur_input_ids[actual_len+1:]
                cur_input_ids = start_ids

                output = self.embedding_model({"input_ids":start_ids.reshape(1, -1).numpy()})
                for k in output:
                    output_value = output[k].squeeze()
                    output_value = output_value[:actual_len]
                    cur_new_input_embeds.append(output_value)

                
                output = self.embedding_model({"input_ids":end_ids.reshape(1, -1).numpy()})
                for k in output:
                    output_value = output[k].squeeze()
                    output_value = output_value[:end_ids.shape[0]]
                    end_new_input_embeds.append(output_value)
                
            cur_new_input_embeds = [torch.from_numpy(x) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)

            end_new_input_embeds = [torch.from_numpy(x) for x in end_new_input_embeds]
            end_new_input_embeds = torch.cat(end_new_input_embeds, dim=0)

            new_input_embeds.append(cur_new_input_embeds)
            end_input_embeds.append(end_new_input_embeds)
            
        new_input_embeds = torch.stack(new_input_embeds, dim=0)
        end_input_embeds = torch.stack(end_input_embeds, dim=0)

        if attention_mask is not None:
            attention_mask = torch.tensor(attention_mask, dtype=torch.float32)
            new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - origin_len), True, dtype=attention_mask.dtype)
            attention_mask[0][origin_len:origin_len+new_input_embeds.shape[1] - origin_len]=new_attn_mask_pad_left[0]

            new_actual_len = new_input_embeds.shape[1]

            new_attention_mask=attention_mask[:,0:512]
            merged_tensor = torch.cat((new_input_embeds, end_input_embeds[:, :self.max_sequence_len
                                                                          -new_input_embeds.shape[1], :]), dim=1)

        return new_attention_mask.numpy(), merged_tensor.numpy(),new_actual_len


prompt = 'Who is the author of this book?\nAnswer the question using a single word or phrase.'
images = ["/home/linaro/workspace/MobileVLM-1688/assets/samples/demo.jpg"]
args = {
    "model_path":"/home/linaro/workspace/MobileVLM-1688/mlir",
    "layers":24,
    "max_seq_len":512,
    "host":"bm1688",##表示需要再什么地方运行，可选bm1688和docker
}
MobileVLM(**args).forward(prompt,images)

