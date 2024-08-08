import os
import numpy as np
from PIL import Image
from utils import get_image_tensor, get_input, get_input_ids, get_output, load_tokenizer,  model_import

from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mobilevlm.utils import tokenizer_image_token, KeywordsStoppingCriteria
import torch
from typing import  Optional

from mobilevlm.conversation import conv_templates, SeparatorStyle

class BmBaseModel:
    def __init__(self,net=None,save_result=True,test=True):
        self.net = net
        self.save_result = save_result
        self.test = test

    def __call__(self, inputs: dict) -> dict:
        # file_path = f"{type(self).__name__}_output.npz"
        # if self.test and os.path.exists(file_path):
        #     return np.load(file_path)
        # else:
        return self.forward(inputs)
        
    def get_input(self,inputs,net):
        if hasattr(self, 'input_impl') and callable(getattr(self, 'input_impl')):
            return self.input_impl(inputs=inputs,net=net)
        else:
            return get_input(inputs,net=net)
    
    def forward(self,inputs:dict)->dict:
        input_shapes = self.get_input(inputs,net=self.net)
        dyn_output_shapes = self.net.forward_dynamic(input_shapes)
        print(f"{type(self).__name__}运行完成")
        outputs = get_output(dyn_output_shapes,self.net)
        if self.save_result:
            print(f"----{type(self).__name__}_output.npz")
            np.savez(f"{type(self).__name__}_output.npz",**outputs)
        return outputs
    
class BmModel(BmBaseModel):
    def __init__(self,pyruntime,model_file,save_result=True,test=True):
        self.model = pyruntime.Model(model_file)
        net = self.model.Net(self.model.networks[0])
        super().__init__(net=net,save_result=save_result,test=test)
    
    def get_model(self):
        return self.model

    
class Embedding(BmBaseModel):
    def __init__(self,model,save_result=True,test=True):
        net = model.get_model().Net("embeding")
        super().__init__(net=net,save_result=save_result,test=test)

    def input_impl(self, inputs, net):
        #数据填充到512
        for key in inputs:
            input = inputs[key]
            padding_width = 512 - input.shape[1]
            inputs[key] = np.pad(input, ((0, 0), (0, padding_width)), 'constant', constant_values=0)
            ## 如果有填充的Id需要，改成0
            inputs[key][inputs[key] < 0] = 0
        return get_input(inputs, net)


class VisionTower(BmBaseModel):
    def __init__(self,model,save_result=True,test=True):
        net = model.get_model().Net("vision_tower")
        super().__init__(net=net,save_result=save_result,test=test)

# class MmProjector(BmBaseModel):
#     def __init__(self,model,save_result=True,test=True):
#         net = model.get_model().Net("mm_projector")
#         super().__init__(net=net,save_result=save_result,test=test)

class MmProjectorMLP(BmBaseModel):
    def __init__(self,model,save_result=True,test=True):
        net = model.get_model().Net("mm_projector_mlp")
        super().__init__(net=net,save_result=save_result,test=test)

class MmProjectorDWN(BmBaseModel):
    def __init__(self,model,save_result=True,test=True):
        net = model.get_model().Net("mm_projector_dwn")
        super().__init__(net=net,save_result=save_result,test=test)

class MmProjectorPEG(BmBaseModel):
    def __init__(self,model,save_result=True,test=True):
        net = model.get_model().Net("mm_projector_peg")
        super().__init__(net=net,save_result=save_result,test=test)

class MmProjector:
    def __init__(self,model,save_result=True,test=True):
        self.mlp = MmProjectorMLP(model,save_result,test)
        self.dwn = MmProjectorDWN(model,save_result,test)
        self.peg = MmProjectorPEG(model,save_result,test)

    def __call__(self, inputs) -> dict:
        return self.forward(inputs)
        

    def forward(self,inputs):
        return self.peg(self.dwn(self.mlp(inputs)))


class Layer(BmBaseModel):
    def __init__(self,model,save_result=True,test=True,index=0):
        self.index = index
        net = model.get_model().Net(f"layer_{index}")
        super().__init__(net=net,save_result=save_result,test=test)

    def __call__(self, inputs: dict) -> dict:
        file_path = f"{type(self).__name__}_{self.index}_output.npz"
        if self.test and os.path.exists(file_path):
            return np.load(file_path)
        else:
            return self.forward(inputs)

class Norm(BmBaseModel):
    def __init__(self,model,save_result=True,test=True):
        net = model.get_model().Net("norm")
        super().__init__(net=net,save_result=save_result,test=test)

class LmHead(BmBaseModel):
    def __init__(self,model,save_result=True,test=True):
        net = model.get_model().Net("lm_head")
        super().__init__(net=net,save_result=save_result,test=test)


class LlamaModel:
    def __init__(self,model,**args):
        self.layers = []
        self.layer_num = args["layers"]
        for i in range(self.layer_num):
            layer_model = Layer(model,test=True,index=i)
            self.layers.append(layer_model)
        self.norm = Norm(model=model,test=True)

    
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
        model_file=os.path.join(self.model_path,"MobileVLM_V2-1.7B-F16.bmodel")
        if args["host"] == "docker":
            self.pyruntime= model_import(model_file=model_file)
        else:
            import importlib
            self.pyruntime = importlib.import_module("bm1688")

        self.all_model = BmModel(pyruntime=self.pyruntime,model_file=model_file,test=False)

        # 模型
        self.embedding_model = Embedding(self.all_model,test=False)
        self.vision_tower_model = VisionTower(self.all_model,test=True)
        self.mm_projector_model = MmProjector(self.all_model,test=True)
        self.lm_head_model = LmHead(self.all_model,test=True)
        self.model_llama = LlamaModel(model=self.all_model,**args)

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
        hidden_states = self.model_llama(inputs_embeds,attention_mask)
        return self.lm_head_model(hidden_states)
        
    def encode_images(self,images):
        output = self.vision_tower_model({"input_images":images})

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
                
                output = self.embedding_model({"0":cur_input_ids[:image_token_start].reshape(1, -1).numpy()})
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

                output = self.embedding_model({"0":start_ids.reshape(1, -1).numpy()})
                for k in output:
                    output_value = output[k].squeeze()
                    output_value = output_value[:actual_len]
                    cur_new_input_embeds.append(output_value)

                
                output = self.embedding_model({"0":end_ids.reshape(1, -1).numpy()})
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

