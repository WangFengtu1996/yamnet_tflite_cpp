/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <functional>
#include <queue>
#include <vector>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "wav_parser.hpp"
// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>


const char * wav_filename = "./miaow_16k.wav";
const string label_filename = "./yamnet_class_map.txt";

TfLiteStatus ReadLabelsFile(const string& file_name,
                            std::vector<string>* result) {
  std::ifstream file(file_name);
  if (!file) {
    std::cerr << "Labels file " << file_name << " not found";
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  // *found_label_count = result->size();
  // const int padding = 16;
  // while (result->size() % padding) {
  //   result->emplace_back();
  // }
  return kTfLiteOk;
}


void get_top_n(float* prediction, int prediction_size,                 std::vector<std::pair<float, int>>* top_results) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>>
      top_result_pq;

  const long count = prediction_size;  // NOLINT(runtime/int)
  float value = 0.0;

  for (int i = 0; i < count; ++i) {
    value = prediction[i];
    top_result_pq.push(std::pair<float, int>(value, i));
  }

  // Copy to output vector and reverse into descending order.
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
}

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  const char* filename = "./lite-model_yamnet_classification_tflite_1.tflite";
  char * wav_file;
  if (argc == 2) {
    // fprintf(stdout, "minimal <tflite model>\n");
     wav_file = argv[1];
    // return 1;
  } else {
    wav_file = const_cast<char*>(wav_filename);
  }
  
  bool verbose = false;
  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

if(verbose)
{
  std::cout << "inputs: " << interpreter->inputs().size() << std::endl;
  int input = interpreter->inputs()[0];
  TfLiteIntArray * input_dims = interpreter->tensor(input)->dims;
  std::cout << "input 0: " << " no" << input << interpreter->GetInputName(0) <<  " dim size : " << input_dims->size <<std::endl;
  std::cout << "dim  " << 1 << " : "<< input_dims->data[1] << std::endl;
  std::cout << "dim  " << 2 << " : "<< input_dims->data[2] << std::endl;
  std::cout << "dim  " << 3 << " : "<< input_dims->data[3] << std::endl;
  for(uint8_t i = 1; i < input_dims->size; i++)
    std::cout << "dim : " << i << " : "<< input_dims->data[i] << std::endl;
  std::cout << "input type : " << (int)interpreter->tensor(input)->type << std::endl;
  std::cout << "outputs: " << interpreter->outputs().size() << std::endl;
  int output = interpreter->outputs()[0];
  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
  std::cout << "output 0 : " << interpreter->GetOutputName(0) <<  " dim size : " << output_dims->size << std::endl;
  for(uint8_t i = 1; i < output_dims->size + 1; i++)
    std::cout << "dim : " << i << " : " << output_dims->data[i] << std::endl;
  std::cout << "output type : " << (int)interpreter->tensor(output)->type << std::endl;

    
  std::cout << "tensors size: " << interpreter->tensors_size() << std::endl;
  std::cout  << "nodes size: " << interpreter->nodes_size() << std::endl ;
  std::cout  << "inputs: " << interpreter->inputs().size()<< std::endl;
  std::cout  << "input(0) name: " << interpreter->GetInputName(0)<< std::endl;

  int t_size = interpreter->tensors_size();
  for (int i = 0; i < t_size; i++) {
    if (interpreter->tensor(i)->name)
      std::cout  << i << ": " << interpreter->tensor(i)->name << ", "
                << interpreter->tensor(i)->bytes << ", "
                << interpreter->tensor(i)->dims->size << ", "
                << interpreter->tensor(i)->type << ", "
                << interpreter->tensor(i)->params.scale << ", "
                << interpreter->tensor(i)->params.zero_point
                << std::endl;
  }

}

  //load wav file
  WAVE wav;
  wav.parse(wav_file);
  auto data_int = wav.getValue();

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());


  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

  int input = interpreter->inputs()[0];
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;

  auto input_infer_data = interpreter->typed_tensor<float>(input);
  int intput_size = interpreter->tensor(input)->bytes / sizeof(float);
  for(int i = 0; i < intput_size; i++ )
    input_infer_data[i] = data_int[i] / 32768.0;

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`

  int output_index = interpreter->outputs()[0];
  auto output_float32 = interpreter->typed_tensor<float>(output_index);
  std::vector<std::string> label_txt;
  std::vector<std::pair<float, int>> top_results;

  if(ReadLabelsFile(label_filename, &label_txt) != kTfLiteOk)
  {
    exit(-1);
  }
  TfLiteIntArray* output_dims = interpreter->tensor(output_index)->dims;
  int output_size = output_dims->data[output_dims->size -1];
  get_top_n(output_float32,output_size,&top_results);
  std::cout << "inference result : " << "class " << label_txt[top_results[0].second] << ", " << "confidence " << top_results[0].first << std::endl;
  return 0;
}
