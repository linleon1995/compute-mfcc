#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <iostream>
#include <fstream>

#include "infer.h"
#include "melspec.h"
#include"wavHeader.h"


char* getBytes(const char* wavPath) {
    // Read the wav header    
    std::ifstream wavFp;
    wavFp.open(wavPath, std::ios::binary);
    wavHeader hdr;
    int headerSize = sizeof(wavHeader);
    wavFp.read((char*)&hdr, headerSize);

    // Initialise buffer
    //get length of file
    wavFp.seekg(0, std::ios::end);
    size_t length = wavFp.tellg();
    wavFp.seekg(0, std::ios::beg);

    //read file
    size_t data_length = length - headerSize;
    char* buffer = new char[data_length];
    wavFp.read(buffer, data_length);

    //delete[] buffer;
    //buffer = nullptr;
    return buffer;
}


loadModelFile::loadModelFile(const char* modelPath) {
    // Convert char* string to a wchar_t* string.
    size_t newsize = strlen(modelPath) + 1;
    wchar_t* modelPath_wc = new wchar_t[newsize];
    size_t convertedChars = 0;
    mbstowcs_s(&convertedChars, modelPath_wc, newsize, modelPath, _TRUNCATE);

    // create session
    session_ = Ort::Session(env, modelPath_wc, Ort::SessionOptions{ nullptr });

    // get input and output name
    Ort::AllocatorWithDefaultOptions ort_alloc;
    char* tmp = session_.GetInputName(0, ort_alloc);
    input_names[0] = _strdup(tmp);
    ort_alloc.Free(tmp);

    tmp = session_.GetOutputName(0, ort_alloc);
    output_names[0] = _strdup(tmp);
    ort_alloc.Free(tmp);
}


std::vector<float> loadModelFile::run_raw(char* bytearray, int byte_len) {
    // preprocess
    std::vector<float> inputData = preprocess(bytearray, byte_len);
    inputShape[2] = height;
    inputShape[3] = width;

    // define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    inputTensor = Ort::Value::CreateTensor<float>(memory_info, inputData.data(), inputData.size(), inputShape.data(), inputShape.size());
    outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), numClasses, outputShape.data(), outputShape.size());

    // run inference
    session_.Run(runOptions, &input_names[0], &inputTensor, 1, &output_names[0], &outputTensor, 1);

    // show scores
    std::vector<float> OutputScores;
    for (size_t i = 0; i < numClasses; ++i) {
        //const auto& result = indexValuePairs[i];
        //std::cout << i << ": " << " " << results[i] << std::endl;
        OutputScores.emplace_back(results[i]);
    }
    return OutputScores;
}


auto loadModelFile::run_and_get_input(char* bytearray, int byte_len) -> std::tuple<std::vector<float>, std::vector<float>> {
    // preprocess
    std::vector<float> inputData = preprocess(bytearray, byte_len);
    inputShape[2] = height;
    inputShape[3] = width;

    // define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    inputTensor = Ort::Value::CreateTensor<float>(memory_info, inputData.data(), inputData.size(), inputShape.data(), inputShape.size());
    outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), numClasses, outputShape.data(), outputShape.size());

    // run inference
    session_.Run(runOptions, &input_names[0], &inputTensor, 1, &output_names[0], &outputTensor, 1);

    // show scores
    std::vector<float> OutputScores;
    for (size_t i = 0; i < numClasses; ++i) {
        //const auto& result = indexValuePairs[i];
        //std::cout << i << ": " << " " << results[i] << std::endl;
        OutputScores.emplace_back(results[i]);
    }
    const std::vector<float> inputData2 = inputData;
    const std::vector<float> OutputScores2 = OutputScores;
    return { inputData2, OutputScores2 };
}



std::vector<float> loadModelFile::run(std::vector<float> inputData) {
    inputShape[2] = 128;
    inputShape[3] = 59;

    // define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    inputTensor = Ort::Value::CreateTensor<float>(memory_info, inputData.data(), inputData.size(), inputShape.data(), inputShape.size());
    outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), numClasses, outputShape.data(), outputShape.size());

    // run inference
    session_.Run(runOptions, &input_names[0], &inputTensor, 1, &output_names[0], &outputTensor, 1);

    // show scores
    std::vector<float> OutputScores;
    for (size_t i = 0; i < numClasses; ++i) {
        //const auto& result = indexValuePairs[i];
        std::cout << i << ": " << " " << results[i] << std::endl;
        OutputScores.emplace_back(results[i]);
    }
    return OutputScores;
}


std::vector<float> loadModelFile::preprocess(char* bytearray, int byte_len) {
    int numCepstra = 12;
    int numFilters = 40;
    int samplingRate = 16000;
    int winLength = 25;
    int frameShift = 10;
    int lowFreq = 50;
    int highFreq = samplingRate / 2;

    // Initialise Mel-Spectrogram class instance
    MelSpec melSpecComputer(samplingRate, numCepstra, winLength, frameShift, numFilters, lowFreq, highFreq);


    // preprocessing
    std::vector<double> inputData = melSpecComputer.processBytes(bytearray, byte_len);
    height = melSpecComputer.height;
    width = melSpecComputer.width;

    std::vector<float> inputDataFloat(inputData.begin(), inputData.end());
    std::vector<float> temp = inputDataFloat;
    inputDataFloat.insert(inputDataFloat.end(), temp.begin(), temp.end());
    inputDataFloat.insert(inputDataFloat.end(), temp.begin(), temp.end());
    return inputDataFloat;
}


int main() {
    // Get model and data reference
    const char* wavPath = "C:/Users/test/Desktop/Leon/Projects/compute-mfcc/data/test/1620231545598_43_36.42_38.42_004.wav";
    const char* modelPath = "C:\\Users\\test\\Desktop\\Leon\\Projects\\Snoring_Detection\\checkpoints\\run_050\\snoring.onnx";

    // Build model
    loadModelFile model(modelPath);

    // Define Input
    std::cout << wavPath << std::endl;
    char* bytearray = getBytes(wavPath);
    int byte_len = 64000;

    //std::vector<float> InputData = preprocess(bytearray, byte_len);
    std::vector<std::vector<std::vector<std::vector<float>>>> InputData_2d = preprocess_2d(bytearray, byte_len);
    //std::vector<float> OutputScores = model.run(InputData);

    //// Inference
    //std::vector<float> OutputScores = model.run(bytearray, byte_len);

    //// Inference and get input data (preprocessed)
    //std::vector<float> InputData_p;
    //std::vector<float> OutputScores_p;
    //std::tie(InputData_p, OutputScores_p) = model.run_and_get_input(bytearray, byte_len);

    return 0;
}