#!/bin/bash
set -e
## 1-model_name
## 2-model_onnx
## 3-input_shapes
## 4-input_types
## 5-mlir
## 6-chip
## 7-output_model
## 8-input_data
## 9-reference_data
##10-no-opt

models=()

transform_model(){
    if [ "${10}" == "no-opt" ];then
        model_transform.py     \
            --model_name $1     \
            --model_def $2     \
            --input_shapes $3     \
            --input_types $4     \
            --mlir $5
    else
        model_transform.py     \
            --model_name $1     \
            --model_def $2     \
            --input_shapes $3     \
            --input_types $4     \
            --mlir $5
            # --test_input $8\
            # --test_result $9\

    fi
    # F16,W8F16也可以
    model_deploy.py \
        --mlir $5 \
        --quantize F16 \
        --chip bm1688 \
        --model $7
        # --test_input $8\
        # --test_reference $9\
}

transform_embedding(){
    echo "转换embedding"
    local model_name="./mlir/embedding.bmodel"
    transform_model "embeding" "$1/embedding.onnx" "[1,512]" "int32" "./mlir/embedding.mlir" bm1688 ${model_name} "./test_case_data/embedding_input.npz" "./test_case_data/embedding_output.npz"
    models+=(${model_name})
}

transform_layers(){
    for i in {0..23}
    do
        echo "开始转换layer_$i"
        local layer_name="./mlir/layer_$i.bmodel"
        transform_model "layer_$i" "$1/layer_$i.onnx" "[[1,512,2048],[1,1,512,512],[1,512]]" "float16" "./mlir/layer_$i.mlir" bm1688 ${layer_name} "./test_case_data/layer_${i}_input.npz" "./test_case_data/layer_${i}_output.npz"
        models+=(${layer_name})
    done
}

transform_norm(){
    echo "开始转换norm"
    local model_name="./mlir/norm.bmodel"
    transform_model "norm" "$1/norm.onnx" "[1,512,2048]" "float16" "./mlir/norm.mlir" bm1688 ${model_name}  "./test_case_data/norm_input.npz" "./test_case_data/norm_output.npz"
    models+=(${model_name})
}

transform_vision_tower(){
    echo "开始转换vision_tower,有精度损失"
    local model_name="./mlir/vision_tower.bmodel"
    transform_model "vision_tower" "$1/vision_tower.onnx" "[1,3,336,336]" "float16" "./mlir/vision_tower.mlir" bm1688 ${model_name} "./test_case_data/vision_tower_input.npz" "./test_case_data/vision_tower_output.npz" "no-opt"
    models+=(${model_name})
}

transform_mm_projector(){
    echo "开始转换mm_projector"
    local model_name="./mlir/mm_projector.bmodel"
    transform_model "mm_projector" "$1/mm_projector.onnx" "[1,576,1024]" "float16" "./mlir/mm_projector.mlir" bm1688 ${model_name} "./test_case_data/mm_projector_input.npz" "./test_case_data/mm_projector_output.npz"
    models+=(${model_name})
}

transform_mm_projector_mlp(){
    echo "开始转换mm_projector_mlp"
    local model_name="./mlir/mm_projector_mlp.bmodel"
    transform_model "mm_projector_mlp" "$1/mm_projector_mlp.onnx" "[1,576,1024]" "float16" "./mlir/mm_projector_mlp.mlir" bm1688 ${model_name} "./test_case_data/mm_projector_mlp_input.npz" "./test_case_data/mm_projector_mlp_output.npz"
    models+=(${model_name})
}

transform_mm_projector_dwn(){
    echo "开始转换mm_projector_dwn"
    local model_name="./mlir/mm_projector_dwn.bmodel"
    transform_model "mm_projector_dwn" "$1/mm_projector_dwn.onnx" "[1,576,2048]" "float16" "./mlir/mm_projector_dwn.mlir" bm1688 ${model_name} "./test_case_data/mm_projector_dwn_input.npz" "./test_case_data/mm_projector_dwn_output.npz"
    models+=(${model_name})
}

transform_mm_projector_peg(){
    echo "开始转换mm_projector_peg"
    local model_name="./mlir/mm_projector_peg.bmodel"
    transform_model "mm_projector_peg" "$1/mm_projector_peg.onnx" "[1,144,2048]" "float16" "./mlir/mm_projector_peg.mlir" bm1688 ${model_name} "./test_case_data/mm_projector_peg_input.npz" "./test_case_data/mm_projector_peg_output.npz"
    models+=(${model_name})
}

transform_lm_head(){
    echo "开始转换lm_head"
    local model_name="./mlir/lm_head.bmodel"
    transform_model "lm_head" "$1/lm_head.onnx" "[1,512,2048]" "float16" "./mlir/lm_head.mlir" bm1688 ${model_name} "./test_case_data/lm_input.npz" "./test_case_data/lm_output.npz"
    models+=(${model_name})
}

together_all_bmodel(){
    echo combine----${models[@]}
    model_tool --combine ${models[@]} -o $1
}

transform() {
    if [ ! -d "mlir" ]; then
        mkdir mlir
    fi
    # transform_embedding $1
    # transform_layers $1
    # transform_norm $1
    transform_vision_tower $1
    # transform_lm_head $1
    # transform_mm_projector_mlp $1
    # transform_mm_projector_dwn $1
    # transform_mm_projector_peg $1
    # together_all_bmodel $2

    echo "模型转换完成🚀🚀🚀"
}

transform "./mobilevlm-tmp/" ./mlir/MobileVLM_V2-1.7B-F16.bmodel
