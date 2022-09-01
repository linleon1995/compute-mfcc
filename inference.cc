#define _CRT_SECURE_NO_DEPRECATE
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <iostream>
#include <fstream>

#include "mfcc_new.cc"
#include "inference.h"



std::vector<int16_t> process_wave(std::ifstream& wavFp) {
    // Read the wav header    
    wavHeader hdr;
    int headerSize = sizeof(wavHeader);
    wavFp.read((char*)&hdr, headerSize);

    //// Check audio format
    //if (hdr.AudioFormat != 1 || hdr.bitsPerSample != 16) {
    //    std::cerr << "Unsupported audio format, use 16 bit PCM Wave" << std::endl;
    //    return 1;
    //}
    //// Check sampling rate
    //if (hdr.SamplesPerSec != fs) {
    //    std::cerr << "Sampling rate mismatch: Found " << hdr.SamplesPerSec << " instead of " << fs << std::endl;
    //    return 1;
    //}

    //// Check sampling rate
    //if (hdr.NumOfChan != 1) {
    //    std::cerr << hdr.NumOfChan << " channel files are unsupported. Use mono." << std::endl;
    //    return 1;
    //}


    // Initialise buffer
    uint16_t bufferLength = 128;
    int16_t* buffer = new int16_t[bufferLength];
    int bufferBPS = (sizeof buffer[0]);

    wavFp.read((char*)buffer, bufferLength * bufferBPS);

    std::vector<int16_t> inputData;
    while (wavFp.gcount() == bufferLength * bufferBPS && !wavFp.eof()) {
        for (int i = 0; i < bufferLength; i++)
            inputData.push_back(buffer[i]);
        wavFp.read((char*)buffer, bufferLength * bufferBPS);
    }

    delete[] buffer;
    buffer = nullptr;
    return inputData;
}


template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}


// A simple option parser
char* getCmdOption(char** begin, char** end, const std::string& value) {
    char** iter = std::find(begin, end, value);
    if (iter != end && ++iter != end)
        return *iter;
    return nullptr;
}


std::vector<int16_t> get_data(const char* wavPath) {
    std::ifstream wavFp;
    wavFp.open(wavPath, std::ios::binary);
    std::vector<int16_t> inputData = process_wave(wavFp);
    return inputData;
}


loadModelFile::loadModelFile(const char* modelPath, const int16_t numClasses) {
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


std::vector<float> loadModelFile::run(std::vector<int16_t> waveform) {
    // preprocess
    std::vector<float> inputData = preprocess(waveform);
    inputShape[2] = height;
    inputShape[3] = width;

    // define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    inputTensor = Ort::Value::CreateTensor<float>(memory_info, inputData.data(), inputData.size(), inputShape.data(), inputShape.size());
    outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), numClasses, outputShape.data(), outputShape.size());

    // run inference
    /*try {
        session_.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
    }
    catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }*/
    session_.Run(runOptions, &input_names[0], &inputTensor, 1, &output_names[0], &outputTensor, 1);

    // sort results
    std::vector<std::pair<size_t, float>> indexValuePairs;
    for (size_t i = 0; i < results.size(); ++i) {
        indexValuePairs.emplace_back(i, results[i]);
    }
    std::sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    // show Top5
    std::vector<float> OutputScores;
    for (size_t i = 0; i < numClasses; ++i) {
        //const auto& result = indexValuePairs[i];
        std::cout << i << ": " << " " << results[i] << std::endl;
        OutputScores.emplace_back(results[i]);
    }

    return OutputScores;
}


std::vector<float> loadModelFile::preprocess(std::vector<int16_t> waveform) {
    int numCepstra = 12;
    int numFilters = 40;
    int samplingRate = 16000;
    int winLength = 25;
    int frameShift = 10;
    int lowFreq = 50;
    int highFreq = samplingRate / 2;

    // Initialise MFCC class instance
    MFCC mfccComputer(samplingRate, numCepstra, winLength, frameShift, numFilters, lowFreq, highFreq);


    // preprocessing
    std::vector<double> inputData = mfccComputer.process_and_return(waveform);
    height = mfccComputer.height;
    width = mfccComputer.width;

    std::vector<float> inputDataFloat(inputData.begin(), inputData.end());
    std::vector<float> temp = inputDataFloat;
    inputDataFloat.insert(inputDataFloat.end(), temp.begin(), temp.end());
    inputDataFloat.insert(inputDataFloat.end(), temp.begin(), temp.end());
    return inputDataFloat;
}


int main() {
    // FIXME:
    const char* wavPath;
    const char* wavPath2;
    const char* wavPath3;
    const char* modelPath;
    const int16_t numClasses = 2;

    /*wavPath = "C:/Users/test/Desktop/Leon/Projects/compute-mfcc/data/1598482996718_21_106.87_108.87_001.wav";
    wavPath2 = "C:/Users/test/Desktop/Leon/Projects/compute-mfcc/data/1606921286802_1_8.93_10.93_001.wav";
    wavPath3 = "C:/Users/test/Desktop/Leon/Projects/compute-mfcc/data/1630779176834_42_4.86_6.86_001.wav";*/
    wavPath = "C:/Users/test/Desktop/Leon/Projects/compute-mfcc/data/test/1620231545598_43_36.42_38.42_004.wav";
    wavPath2 = "C:/Users/test/Desktop/Leon/Projects/compute-mfcc/data/test/1630681292279_88_10.40_12.40_003.wav";
    wavPath3 = "C:/Users/test/Desktop/Leon/Projects/compute-mfcc/data/test/1630779176834_42_4.86_6.86_001.wav";
    modelPath = "C:\\Users\\test\\Desktop\\Leon\\Projects\\Snoring_Detection\\checkpoints\\run_050\\snoring.onnx";

    
    loadModelFile model(modelPath, numClasses);

    std::cout << wavPath << std::endl;
    std::vector<int16_t> waveform = get_data(wavPath);
    std::vector<float> OutputScores = model.run(waveform);

    std::cout << wavPath2 << std::endl;
    std::vector<int16_t> waveform2 = get_data(wavPath2);
    std::vector<float> OutputScores2 = model.run(waveform2);

    std::cout << wavPath3 << std::endl;
    std::vector<int16_t> waveform3 = get_data(wavPath3);
    std::vector<float> OutputScores3 = model.run(waveform3);
    return 0;
}