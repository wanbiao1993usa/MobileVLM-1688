//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#define USE_FFMPEG 1
#define USE_OPENCV 1
#define USE_BMCV 1
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <time.h>
#include <memory>
#include <getopt.h>
#include "cnpy.h"
#include "Log.h"
#include <sail/cvwrapper.h>
#include <sail/tensor.h>
#include <sail/engine.h>

class Model{
    public:
    Model(int device_id,std::string input_file,std::string model_file,std::string output_file):
            device_id(device_id),input_file(input_file),model_file(model_file),output_file(output_file){
                initModel();
                getInput();
                process();
                putOutput();
            }

    private:
    void initModel(){
        mode = sail::SYSIO;
        handle = std::make_shared<sail::Handle>(device_id);
        engine = std::make_shared<sail::Engine>(model_file, *handle.get(), mode);

        graph_name = engine->get_graph_names()[0];
        input_names = engine->get_input_names(graph_name);
        output_names = engine->get_output_names(graph_name);

        for(int i=0;i<input_names.size();i++){
            input_shape.push_back(engine->get_input_shape(graph_name,input_names[i]));
            input_dtype.push_back(engine->get_input_dtype(graph_name,input_names[i]));
            input_scale.push_back(engine->get_input_scale(graph_name,input_names[i]));
            input_tensors.push_back(std::make_unique<sail::Tensor>(*handle.get(),input_shape[i],input_dtype[i],1,1));
            LOG("inputname:",input_names[i],"\n");
            LOG("shape:");printVecotr(input_shape[i]);LOG("\n");
        }
        LOG("dtype:");printVecotr(input_dtype);LOG("\n");
        LOG("scale:");printVecotr(input_scale);LOG("\n");

        for (int i = 0; i < output_names.size(); i++) {
            output_shape.push_back(engine->get_output_shape(graph_name, output_names[i]));
            output_dtype.push_back(engine->get_output_dtype(graph_name, output_names[i]));
            output_scale.push_back(engine->get_output_scale(graph_name, output_names[i]));
            output_tensors.push_back(std::make_unique<sail::Tensor>(*handle.get(), output_shape[i], output_dtype[i], 1,1));
            LOG("outputname:",output_names[i],"\n");
            LOG("shape:");printVecotr(output_shape[i]);LOG("\n");
        }
        LOG("dtype:");printVecotr(output_dtype);LOG("\n");
        LOG("scale:");printVecotr(output_scale);LOG("\n");
    }
    void getInput(){
        cnpy::npz_t input_npz = cnpy::npz_load(input_file);
        std::vector<std::string> names;
        std::vector<cnpy::NpyArray> values;
        // 列出 npz 文件中的所有数组
        for (auto it = input_npz.begin(); it != input_npz.end(); ++it) {
            LOG("name",it->first);
            names.push_back(it->first);
            values.push_back(input_npz[it->first]);
        }

        for(int index=0;index<values.size();index++){
            LOG("input_name:",names[index],"\n");
            switch (input_dtype[index])
            {
            case bm_data_type_t::BM_FLOAT32:
                    printNpyArray<float>(values[index]);
                break;
            case bm_data_type_t::BM_INT32:
                    printNpyArray<int32_t>(values[index]);
                break;
            
            case bm_data_type_t::BM_INT8:
                printNpyArray<int8_t>(values[index]);
            break;
            case bm_data_type_t::BM_UINT8:
                printNpyArray<uint8_t>(values[index]);
            break;
            case bm_data_type_t::BM_UINT32:
                printNpyArray<uint32_t>(values[index]);
            break;
            case bm_data_type_t::BM_FLOAT16:
            case bm_data_type_t::BM_BFLOAT16:
            case bm_data_type_t::BM_INT16:
            case bm_data_type_t::BM_UINT16:
                printNpyArray<uint32_t>(values[index]);
            break;
            default:
                    LOG("nor support");
                break;
            }
        }

        input =engine->create_input_tensors_map(graph_name, -1);
        for(int i=0;i<input_tensors.size();i++){
            auto input_tensor = input_tensors[i].get();
            auto shape = input_shape[i];
            // input_tensor->reset_sys_data(input_npz[input_names[i]].data<void>(), shape);
            LOG("input_name",input_names[i],"\n");
            LOG("length",getTensorLength(shape),"\n");
            switch (input_dtype[i])
            {
                case bm_data_type_t::BM_FLOAT32:{
                    LOG("float32\n");
                    input_tensor->reset_sys_data(input_npz[input_names[i]].data<float>(), shape);
                }break;
                case bm_data_type_t::BM_INT32:{
                    LOG("int32\n");
                    input_tensor->reset_sys_data(input_npz[input_names[i]].data<int32_t>(), shape);
                }break;
                
                case bm_data_type_t::BM_INT8:{
                    LOG("int8\n");
                    input_tensor->reset_sys_data(input_npz[input_names[i]].data<int8_t>(), shape);
                }break;
                case bm_data_type_t::BM_UINT8:{
                    LOG("uint8\n");
                    input_tensor->reset_sys_data(input_npz[input_names[i]].data<uint8_t>(), shape);
                }break;
                case bm_data_type_t::BM_UINT32:{
                    LOG("uint32\n"); 
                    input_tensor->reset_sys_data(input_npz[input_names[i]].data<uint32_t>(), shape);
                }break;
                case bm_data_type_t::BM_FLOAT16:
                case bm_data_type_t::BM_BFLOAT16:
                case bm_data_type_t::BM_INT16:
                case bm_data_type_t::BM_UINT16:{
                    input_tensor->reset_sys_data(input_npz[input_names[i]].data<uint32_t>(), shape);
                }break;
                default:{
                    LOG("nor support");
                }  
                break;
            }
            input_tensor->sync_s2d();
            input[input_names[i]] = input_tensor;
        }
        

    }
    void process(){
        output = engine->create_output_tensors_map(graph_name,1);
        for (int i = 0; i < output_names.size(); i++) {
            output[output_names[i]] = output_tensors[i].get();
        }
        engine->process(graph_name, input, output);
    }
    void putOutput(){
        for(auto i=output.begin();i!=output.end();i++){
            LOG(i->first);
            sail::Tensor * value = i->second;
            void * data = value->sys_data();
            auto index = std::distance(output.begin(), i);
            auto &cur_shape = output_shape[index];
            std::vector<size_t> shape(cur_shape.size());
            for(int k=0;k<shape.size();k++){
                shape[k] = static_cast<size_t>(cur_shape[k]);
            }
            switch (output_dtype[index])
            {
                case bm_data_type_t::BM_FLOAT32:
                    printAndSave<float>(static_cast<float*>(data),shape);
                    if(i == output.begin()){
                        cnpy::npz_save(output_file, i->first, static_cast<float*>(data), shape, "w");
                    }else{
                        cnpy::npz_save(output_file, i->first, static_cast<float*>(data), shape, "a");
                    }
                break;
                case bm_data_type_t::BM_INT8:
                    printAndSave<int8_t>(static_cast<int8_t*>(data),shape);
                    if(i == output.begin()){
                        cnpy::npz_save(output_file, i->first, static_cast<int8_t*>(data), shape, "w");
                    }else{
                        cnpy::npz_save(output_file, i->first, static_cast<int8_t*>(data), shape, "a");
                    }
                break;
                case bm_data_type_t::BM_UINT8:
                    printAndSave<uint8_t>(static_cast<uint8_t*>(data),shape);
                    if(i == output.begin()){
                        cnpy::npz_save(output_file, i->first, static_cast<uint8_t*>(data), shape, "w");
                    }else{
                        cnpy::npz_save(output_file, i->first, static_cast<uint8_t*>(data), shape, "a");
                    }
                break;
                case bm_data_type_t::BM_INT32:
                    printAndSave<int32_t>(static_cast<int32_t*>(data),shape);
                    if(i == output.begin()){
                        cnpy::npz_save(output_file, i->first, static_cast<int32_t*>(data), shape, "w");
                    }else{
                        cnpy::npz_save(output_file, i->first, static_cast<int32_t*>(data), shape, "a");
                    }
                break;
                case bm_data_type_t::BM_UINT32:
                    printAndSave<uint32_t>(static_cast<uint32_t*>(data),shape);
                    if(i == output.begin()){
                        cnpy::npz_save(output_file, i->first, static_cast<uint32_t*>(data), shape, "w");
                    }else{
                        cnpy::npz_save(output_file, i->first, static_cast<uint32_t*>(data), shape, "a");
                    }
                break;
                // case bm_data_type_t::BM_FLOAT16:
                // case bm_data_type_t::BM_BFLOAT16:
                // case bm_data_type_t::BM_INT16:
                // case bm_data_type_t::BM_UINT16:
                //     printAndSave<uint32_t>(static_cast<uint32_t*>(data),shape);
                //     if(i == output.begin()){
                //         cnpy::npz_save(output_file, i->first, static_cast<uint32_t*>(data), shape, "w");
                //     }else{
                //         cnpy::npz_save(output_file, i->first, static_cast<uint32_t*>(data), shape, "a");
                //     }
                //     break;
                default:
                    LOG("nor support");
                break;
            }
        }
    }

    template<typename T>
    void printNpyArray(cnpy::NpyArray arr){
        
        T* data = arr.data<T>();
        // 输出形状
        LOG("shape:");
        int32_t all_data_length = 1;
        for (auto d:arr.shape){
            LOG(d);
            all_data_length = all_data_length *d;
        }
        LOG("all_data_length",all_data_length,"\n");
        LOG("\n");
        LOG("data:");
        if(all_data_length > 50){
            for(int i=0;i<25;i++){
                LOG(data[i]);
            }
            LOG("....");
            for(int i=all_data_length-25;i<all_data_length;i++){
                LOG(data[i]);
            }
        }else{
            // 输出数组数据（假设数组是1维的）
            for (size_t i = 0; i < all_data_length; ++i) {
                LOG(data[i]," ");
            }
        }
        
        LOG("\n");
    }

    template<typename T>
    void printVecotr(std::vector<T>& data){
        for(auto &d:data){
            LOG(d);
        }
        LOG("\n");
    }

    template<typename K,typename T>
    void printAndSave(K* data,std::vector<T> shape){
        auto all_length = getTensorLength(shape);
        if(all_length > 50){
            for(int i=0;i<25;i++){
                LOG(data[i]);
            }
            LOG(".....");
            for(int i=all_length-25;i<all_length;i++){
                LOG(data[i]);
            }
        }else{
            for(int i=0;i<all_length;i++){
                LOG(data[i]);
            }
        }
    }

    template<typename T>
    int getTensorLength(std::vector<T> shape){
        auto all_length = 1;
        for(auto &s:shape){
            all_length = all_length * s;
        }
        return all_length;
    }



    private:
    std::string input_file,model_file,output_file;
    int device_id= 0;
    sail::IOMode mode;
    std::shared_ptr<sail::Handle> handle;
    std::shared_ptr<sail::Engine> engine;

    std::string graph_name;
    std::vector<string> input_names;
    std::vector<string> output_names;

    std::vector<std::vector<int>> input_shape;
    std::vector<bm_data_type_t> input_dtype;
    std::vector<float> input_scale;

    std::vector<std::vector<int>> output_shape;
    std::vector<bm_data_type_t> output_dtype;
    std::vector<float> output_scale;

    std::vector<std::unique_ptr<sail::Tensor>> input_tensors;
    std::vector<std::unique_ptr<sail::Tensor>> output_tensors;

    std::map<std::string, sail::Tensor *> input;
    std::map<std::string, sail::Tensor *> output;
};


int main(int argc, char* argv[]) {
    LOG("start process\n");
    bool debug = true;
    struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"model", required_argument, 0, 'm'},
        {"output", required_argument, 0, 'o'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    int opt;
    int option_index = 0;

    std::string input_file,model_file,output_file;

    while ((opt = getopt_long(argc, argv, "hi:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                input_file = optarg;
                break;
            case 'm':
                model_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'h':
                std::cout << "Usage: " << argv[0] << " [--input arg] [--help]\n";
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [--input arg] [--help]\n";
                return 1;
        }
    }
    LOG("input:",input_file," output:",output_file," modelfile:",model_file,"\n");
    Model(0,input_file,model_file,output_file);

    return 0;
}