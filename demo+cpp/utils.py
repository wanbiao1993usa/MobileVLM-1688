import numpy as np
import struct
from PIL import Image
from mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import torch
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    resize,
    to_channel_dimension_format,
)
from transformers.image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    resize,
    to_channel_dimension_format,
)
from transformers.image_transforms import center_crop, normalize, rescale
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from typing import Dict, List, Optional, Union
import PIL
from mobilevlm.conversation import conv_templates, SeparatorStyle
import os
import importlib
from transformers import AutoTokenizer
from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

equal_dtypes = {
    "i4": "int4",
    "u4": "uint4",
    "ui4": "uint4",
    "i8": "int8",
    "u8": "uint8",
    "ui8": "uint8",
    "si8": "int8",
    "i16": "int16",
    "u16": "uint16",
    "i32": "int32",
    "i64": "int64",
    "ui16": "uint16",
    "i32": "int32",
    "ui32": "uint32",
    "f16": "float16",
    "f32": "float32",
}


def lowering(input, pdtype, pshape, pzero_point=0, pscale=1):
    if equal_dtypes.get(pdtype, pdtype) == input.dtype.name:
        res = input.reshape(pshape)
    elif pdtype == "i8" and input.dtype == np.float32:
        data = round_away_from_zero(input * pscale + pzero_point)
        res = np.clip(data, -128, 127).astype(np.int8).reshape(pshape)
    elif pdtype == "i8" and input.dtype == np.int32:
        data = round_away_from_zero(input * pscale + pzero_point)
        res = np.clip(data, -128, 127).astype(np.int8).reshape(pshape)
    elif pdtype == "u8" and input.dtype == np.int32:
        data = round_away_from_zero(input * pscale + pzero_point)
        res = np.clip(data, 0, 255).astype(np.int8).reshape(pshape)
    elif pdtype == "u8" and input.dtype == np.float32:
        data = round_away_from_zero(input * pscale + pzero_point)
        res = np.clip(data, 0, 255).astype(np.uint8).reshape(pshape)
    elif pdtype == "u16" and (input.dtype == np.float32 or input.dtype == np.int32):
        res = input.astype(np.uint16).reshape(pshape)
    elif pdtype == "i16" and (input.dtype == np.float32 or input.dtype == np.int32):
        res = input.astype(np.int16).reshape(pshape)
    elif pdtype == "f16" and input.dtype == np.float32:
        res = input.astype(np.float16)
    elif pdtype == "bf16" and input.dtype == np.float32:
        res = fp32_to_bf16(input).reshape(pshape)
    elif pdtype == "i32" and (input.dtype == np.float32 or input.dtype == np.int64):
        res = input.astype(np.int32).reshape(pshape)
    elif pdtype == "u32" and (input.dtype == np.float32 or input.dtype == np.int64 or input.dtype == np.uint32):
        res = input.astype(np.uint32).reshape(pshape)
    elif pdtype == "i4" and input.dtype == np.float32:
        data = round_away_from_zero(input * pscale + pzero_point)
        res = np.clip(data, -8, 7).astype(np.int8).reshape(pshape)
    elif pdtype == "u4" and input.dtype == np.float32:
        data = round_away_from_zero(input * pscale + pzero_point)
        res = np.clip(data, 0, 15).astype(np.uint8).reshape(pshape)
    elif pdtype == "f32":
        res = input.astype(np.float32).reshape(pshape)
    else:
        raise ValueError(f"unknown type: form {input.dtype} to {pdtype}")
    return res


def round_away_from_zero(x):
    a = np.floor(np.abs(x) + 0.5)
    return np.sign(x) * a


def bf16_to_fp32(d_bf16):
    s = d_bf16.shape
    d_bf16 = d_bf16.flatten()
    assert d_bf16.dtype == np.uint16
    d_fp32 = np.empty_like(d_bf16, dtype=np.float32)
    for i in range(len(d_bf16)):
        d_fp32[i] = struct.unpack("<f", struct.pack("<HH", 0, d_bf16[i]))[0]
    return d_fp32.reshape(s)


def fp32_to_bf16(d_fp32):
    s = d_fp32.shape
    d_fp32 = d_fp32.flatten()
    assert d_fp32.dtype == np.float32
    d_bf16 = np.empty_like(d_fp32, dtype=np.uint16)
    for i in range(len(d_bf16)):
        bytes = struct.pack("f", d_fp32[i])
        d_bf16[i] = struct.unpack("<H", struct.pack("BB", bytes[2], bytes[3]))[0]
    return d_bf16.reshape(s)

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def resize_custom(
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        size = get_size_dict(size, default_to_square=False)
        if "shortest_edge" not in size:
            raise ValueError(f"The `size` parameter must contain the key `shortest_edge`. Got {size.keys()}")
        output_size = get_resize_output_image_size(
            image, size=size["shortest_edge"], default_to_square=False, input_data_format=input_data_format
        )
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

def center_crop_custom(
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The size dictionary must have keys 'height' and 'width'. Got {size.keys()}")
        return center_crop(
            image,
            size=(size["height"], size["width"]),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

def rescale_custom(
        image: np.ndarray,
        scale: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        return rescale(image, scale=scale, data_format=data_format, input_data_format=input_data_format, **kwargs)


def normalize_custom(
        image: np.ndarray,
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        return normalize(
            image, mean=mean, std=std, data_format=data_format, input_data_format=input_data_format, **kwargs
        )



def preprocess(images: ImageInput,data_format: Optional[ChannelDimension] = ChannelDimension.FIRST) -> PIL.Image.Image:
    do_resize = True
    size = {'shortest_edge': 336}
    resample = 3
    do_center_crop = True
    crop_size = {'height': 336, 'width': 336}
    do_rescale = True
    rescale_factor = 0.00392156862745098
    do_normalize = True
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    do_convert_rgb = True

    images = make_list_of_images(images)

    if not valid_images(images):
        raise ValueError(
            "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
            "torch.Tensor, tf.Tensor or jax.ndarray."
        )

    if do_resize and size is None:
        raise ValueError("Size must be specified if do_resize is True.")

    if do_center_crop and crop_size is None:
        raise ValueError("Crop size must be specified if do_center_crop is True.")

    if do_rescale and rescale_factor is None:
        raise ValueError("Rescale factor must be specified if do_rescale is True.")

    if do_normalize and (image_mean is None or image_std is None):
        raise ValueError("Image mean and std must be specified if do_normalize is True.")

    # PIL RGBA images are converted to RGB
    if do_convert_rgb:
        images = [convert_to_rgb(image) for image in images]

    # All transformations expect numpy arrays.
    images = [to_numpy_array(image) for image in images]

    # We assume that all images have the same channel dimension format.
    input_data_format = infer_channel_dimension_format(images[0])

    if do_resize:
        images = [
            resize_custom(image=image, size=size, resample=resample, input_data_format=input_data_format)
            for image in images
        ]

    if do_center_crop:
        images = [
            center_crop_custom(image=image, size=crop_size, input_data_format=input_data_format) for image in images
        ]

    if do_rescale:
        images = [
            rescale_custom(image=image, scale=rescale_factor, input_data_format=input_data_format)
            for image in images
        ]

    if do_normalize:
        images = [
            normalize_custom(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
            for image in images
        ]

    images = [
        to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
    ]

    data = {"pixel_values": images}
    return BatchFeature(data=data, tensor_type="pt")

def get_chip_from_model(model_file: str) -> str:
    fd = os.popen("model_tool --chip {}".format(model_file))
    chip = fd.read()
    fd.close()
    return chip

def get_chip_so(chip:str)->str:
    lib_so = 'libcmodel_1684x.so'
    if chip == 'BM1688' or chip == 'CV186X':
        lib_so = 'libcmodel_1688.so'
    elif chip == 'BM1684':
        lib_so = 'libcmodel_1684.so'
    elif chip == "BM1690":
        lib_so = 'libcmodel_bm1690.so'
    elif chip == "MARS3":
        lib_so = 'libcmodel_mars3.so'
    return lib_so

def get_custom_so(chip:str)->str:
    lib_so = 'libcmodel_custom_1684x.so'
    if chip == 'BM1688':
        lib_so = 'libcmodel_custom_1688.so'
    # elif chip == 'BM1684':
    #     lib_so = 'libcmodel_custom_1684.so'
    # elif chip == "BM1690":
    #     lib_so = 'libcmodel_custom_bm1690.so'
    # elif chip == "MARS3":
    #     lib_so = 'libcmodel_custom_mars3.so'
    return lib_so


def create_chip_link(model_file: str):
    chip = get_chip_from_model(model_file)
    lib_so = get_chip_so(chip)
    assert(os.path.exists("{}/lib/{}".format(os.getenv("TPUC_ROOT"), lib_so)))
    cmd = 'ln -sf $TPUC_ROOT/lib/{} $TPUC_ROOT/lib/libcmodel.so'.format(lib_so)
    os.system(cmd)

def create_custom_link(model_file:str):
    chip=get_chip_from_model(model_file)
    lib_so = get_custom_so(chip)
    if os.path.exists("{}/lib/{}".format(os.getenv("TPUC_ROOT"), lib_so)):
        cmd = 'ln -sf $TPUC_ROOT/lib/{} $TPUC_ROOT/lib/libcmodel_custom.so'.format(lib_so)
        os.system(cmd)

def model_import(model_file: str):
    create_chip_link(model_file)
    create_custom_link(model_file)
    pyruntime = importlib.import_module("pyruntime_bm")
    print(pyruntime)
    return pyruntime

def get_input(inputs:dict,net):
    ## 准备输入数据
    input_shapes = []
    only_one = len(inputs) == 1
    for i in net.inputs:
        if not only_one:
            assert i.name in inputs
            input = inputs[i.name]
        else:
            input = list(inputs.values())[0]

        ##判断形状
        overflow = np.prod(i.data.shape) - np.prod(input.shape)
        min_len = 0
        if overflow > 0:
            input_shapes.append(input.shape)
            min_len = np.prod(input.shape)
        else:
            input_shapes.append(i.data.shape)
            min_len = np.prod(i.data.shape)
        
        i.data.reshape(-1)[:min_len] = (lowering(input, pdtype=i.dtype, pshape=input.shape, pzero_point=i.qzero_point, pscale=i.qscale).flatten()[:min_len])
    return input_shapes

def get_output(dyn_output_shapes,net):
    outputs = dict()
    dyn_idx = 0
    for i in net.outputs:
        if (i.dtype == 'u16'):
            outputs[i.name] = np.array(i.data.astype(np.float32))
        elif (i.dtype == "f16"):
            outputs[i.name] = np.array(i.data.astype(np.float32))
        elif (i.dtype == "bf16"):
            outputs[i.name] = bf16_to_fp32(i.data)
        else:
            outputs[i.name] = np.array(i.data)
        
        if outputs[i.name].shape != dyn_output_shapes[dyn_idx]:
            dyn_len = np.prod(dyn_output_shapes[dyn_idx])
            outputs[i.name] = outputs[i.name].flatten()[:dyn_len].reshape(
                *dyn_output_shapes[dyn_idx])
            dyn_idx += 1
    return outputs

def load_tokenizer(tokenizer_path):
    return AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

def get_input_ids(tokenizer,prompt):
    return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)



def get_image_tensor(images):
    image_mean=[0.48145466,0.4578275,0.40821073]
    new_images = []
    for image in images:
        image = expand2square(image, tuple(int(x*255) for x in image_mean))
        image = preprocess(image)['pixel_values'][0]
        new_images.append(image)

    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images.numpy()