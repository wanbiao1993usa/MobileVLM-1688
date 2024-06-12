##转换embed层
model_transform.py     \
    --model_name embedding     \
    --model_def ./tmp/embedding.onnx     \
    --input_shapes [2048]     \
    --input_types "int32"     \
    --mlir ./tmp/embedding.mlir

model_deploy.py \
    --mlir ./tmp/embedding.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model embedding_f16.bmodel

##转换中间layer层
for i in {0..23}
do
model_transform.py \
    --model_name layer_$i \
    --model_def ./tmp/layer_$i.onnx \
    --mlir ./tmp/layer_$i.mlir

model_deploy.py \
    --mlir ./tmp/layer_$i.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model ./tmp/layer_$i.bmodel

models=${models}' ''./tmp/layer_'$i'.bmodel'
done

echo $models

##转换Norm
model_transform.py \
    --model_name norm \
    --model_def ./tmp/norm.onnx \
    --mlir ./tmp/norm.mlir

model_deploy.py \
    --mlir ./tmp/norm.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model ./tmp/norm.bmodel

##vision_tower
model_transform.py \
    --model_name vision_tower \
    --model_def ./tmp/vision_tower.onnx \
    --mlir ./tmp/vision_tower.mlir

model_deploy.py \
    --mlir ./tmp/vision_tower.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model ./tmp/vision_tower.bmodel

##mm_projector
model_transform.py \
    --model_name mm_projector \
    --model_def ./tmp/mm_projector.onnx \
    --mlir ./tmp/mm_projector.mlir

model_deploy.py \
    --mlir ./tmp/mm_projector.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model ./tmp/mm_projector.bmodel

##lm_head
model_tool --combine $models -o ./tmp/MobileVLM_V2-1.7B-F16.bmodel

echo "模型转换完成🚀🚀🚀"

