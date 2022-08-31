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
    /*constexpr int64_t numChannels = 3;
    constexpr int64_t height = 59;
    constexpr int64_t width = 128;
    constexpr int64_t numClasses = 2;
    constexpr int64_t numInputElements = numChannels * height * width;*/

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

    // get input shape
    Ort::TypeInfo info = session_.GetInputTypeInfo(0);
    auto tensor_info = info.GetTensorTypeAndShapeInfo();
    size_t dim_count = tensor_info.GetDimensionsCount();
    std::vector<int64_t> dims(dim_count);
    tensor_info.GetDimensions(dims.data(), dims.size());
    /*const int64_t channels_ = dims[1];
    const int64_t height_ = dims[2];
    const int64_t width_ = dims[3];*/
    const int64_t channels_ = 3;
    const int64_t height_ = 59;
    const int64_t width_ = 128;
    inputShape[0] = 1;
    inputShape[1] = channels_;
    inputShape[2] = height_;
    inputShape[3] = width_;

    //FIXME:
    constexpr int64_t numInputElements = 59 * 128 * 3;
    constexpr int64_t numClass = 2;

    // define shape
    //inputShape = { 1, channels_, height_, width_ };
    outputShape[0] = 1;
    outputShape[1] = 2;

    //// define array
    //std::array<float, numInputElements> input;
    //std::array<float, numClass> results;

    // define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), numInputElements, inputShape.data(), inputShape.size());
    outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), numClass, outputShape.data(), outputShape.size());

    // define names
    //Ort::AllocatorWithDefaultOptions ort_alloc;
    char* inputName = session_.GetInputName(0, ort_alloc);
    char* outputName = session_.GetOutputName(0, ort_alloc);
    const std::array<const char*, 1> inputNames = { inputName };
    const std::array<const char*, 1> outputNames = { outputName };
    ort_alloc.Free(inputName);
    ort_alloc.Free(outputName);


}


std::vector<float> loadModelFile::run(std::vector<int16_t> waveform) {
    std::vector<float> inputData = preprocess(waveform);

    // copy image data to input array
    //std::copy(inputData.begin(), inputData.end(), std::back_inserter(input));
    std::copy(inputData.begin(), inputData.end(), input.begin());
    //input.assign(inputData.begin(), inputData.end());

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
    for (size_t i = 0; i < 2; ++i) {
        const auto& result = indexValuePairs[i];
        std::cout << i << ": " << " " << result.second << std::endl;
    }

    std::vector<float> OutputScores;
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
    std::vector<float> inputDataFloat(inputData.begin(), inputData.end());

    std::vector<float> temp = inputDataFloat;
    inputDataFloat.insert(inputDataFloat.end(), temp.begin(), temp.end());
    inputDataFloat.insert(inputDataFloat.end(), temp.begin(), temp.end());
    return inputDataFloat;
}


int onnx_inference(std::vector<float> inputData, const char* modelPath) {
    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session(nullptr);

    constexpr int64_t numChannels = 3;
    constexpr int64_t height = 59;
    constexpr int64_t width = 128;
    constexpr int64_t numClasses = 2;
    constexpr int64_t numInputElements = numChannels * height * width;

    // Convert char* string to a wchar_t* string.
    size_t newsize = strlen(modelPath) + 1;
    wchar_t* modelPath_wc = new wchar_t[newsize];
    size_t convertedChars = 0;
    mbstowcs_s(&convertedChars, modelPath_wc, newsize, modelPath, _TRUNCATE);

    // create session
    session = Ort::Session(env, modelPath_wc, Ort::SessionOptions{ nullptr });

    // define shape
    const std::array<int64_t, 4> inputShape = { 1, numChannels, height, width };
    const std::array<int64_t, 2> outputShape = { 1, numClasses };

    // define array
    std::array<float, numInputElements> input;
    std::array<float, numClasses> results;

    // define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

    // copy image data to input array
    copy(inputData.begin(), inputData.end(), input.begin());



    // define names
    Ort::AllocatorWithDefaultOptions ort_alloc;
    char* inputName = session.GetInputName(0, ort_alloc);
    char* outputName = session.GetOutputName(0, ort_alloc);
    const std::array<const char*, 1> inputNames = { inputName };
    const std::array<const char*, 1> outputNames = { outputName };
    ort_alloc.Free(inputName);
    ort_alloc.Free(outputName);


    // run inference
    try {
        session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
    }
    catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    // sort results
    std::vector<std::pair<size_t, float>> indexValuePairs;
    for (size_t i = 0; i < results.size(); ++i) {
        indexValuePairs.emplace_back(i, results[i]);
    }
    std::sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    // show Top5
    for (size_t i = 0; i < 2; ++i) {
        const auto& result = indexValuePairs[i];
        std::cout << i << ": " << " " << result.second << std::endl;
    }
}


//// Process each file
//std::vector <double> processFileData(MFCC& mfccComputer, const char* wavPath) {
//    // Initialise input and output streams    
//    std::ifstream wavFp;
//    wavFp.open(wavPath, std::ios::binary);
//    std::vector<double> inputData = mfccComputer.process_and_return(wavFp);
//
//    wavFp.close();
//    return inputData;
//}

//
//int main(int argc, char* argv[]) {
//    std::string USAGE = "compute-mfcc : MFCC Extractor\n";
//    USAGE += "OPTIONS\n";
//    USAGE += "--input           : Input 16 bit PCM Wave file\n";
//    USAGE += "--model           : Input onnx model file\n";
//    USAGE += "--numcepstra      : Number of output cepstra, excluding log-energy (default=12)\n";
//    USAGE += "--numfilters      : Number of Mel warped filters in filterbank (default=40)\n";
//    USAGE += "--samplingrate    : Sampling rate in Hertz (default=16000)\n";
//    USAGE += "--winlength       : Length of analysis window in milliseconds (default=25)\n";
//    USAGE += "--frameshift      : Frame shift in milliseconds (default=10)\n";
//    USAGE += "--lowfreq         : Filterbank low frequency cutoff in Hertz (default=50)\n";
//    USAGE += "--highfreq        : Filterbank high freqency cutoff in Hertz (default=samplingrate/2)\n";
//    USAGE += "USAGE EXAMPLES\n";
//    USAGE += "compute-mfcc --input input.wav --output output.mfc\n";
//    USAGE += "compute-mfcc --input input.wav --output output.mfc --samplingrate 8000\n";
//    USAGE += "compute-mfcc --inputlist input.list --outputlist output.list\n";
//    USAGE += "compute-mfcc --inputlist input.list --outputlist output.list --numcepstra 17 --samplingrate 44100\n";
//
//    const char* wavPath = getCmdOption(argv, argv + argc, "--input");
//    const char* modelPath = getCmdOption(argv, argv + argc, "--model");
//    char* numCepstraC = getCmdOption(argv, argv + argc, "--numcepstra");
//    char* numFiltersC = getCmdOption(argv, argv + argc, "--numfilters");
//    char* samplingRateC = getCmdOption(argv, argv + argc, "--samplingrate");
//    char* winLengthC = getCmdOption(argv, argv + argc, "--winlength");
//    char* frameShiftC = getCmdOption(argv, argv + argc, "--frameshift");
//    char* lowFreqC = getCmdOption(argv, argv + argc, "--lowfreq");
//    char* highFreqC = getCmdOption(argv, argv + argc, "--highfreq");
//
//    // FIXME:
//    //wavPath = "C:/Users/test/Desktop/Leon/Projects/compute-mfcc/data/1598482996718_21_106.87_108.87_001.wav";
//    //wavPath = "C:/Users/test/Desktop/Leon/Projects/compute-mfcc/data/1606921286802_1_8.93_10.93_001.wav";
//    wavPath = "C:/Users/test/Desktop/Leon/Projects/compute-mfcc/data/1630779176834_42_4.86_6.86_001.wav";
//    modelPath = "C:\\Users\\test\\Desktop\\Leon\\Projects\\Snoring_Detection\\checkpoints\\run_050\\snoring.onnx";
//
//
//    // Assign variables
//    int numCepstra = (numCepstraC ? atoi(numCepstraC) : 12);
//    int numFilters = (numFiltersC ? atoi(numFiltersC) : 40);
//    int samplingRate = (samplingRateC ? atoi(samplingRateC) : 16000);
//    int winLength = (winLengthC ? atoi(winLengthC) : 25);
//    int frameShift = (frameShiftC ? atoi(frameShiftC) : 10);
//    int lowFreq = (lowFreqC ? atoi(lowFreqC) : 50);
//    int highFreq = (highFreqC ? atoi(highFreqC) : samplingRate / 2);
//
//    // Initialise MFCC class instance
//    MFCC mfccComputer(samplingRate, numCepstra, winLength, frameShift, numFilters, lowFreq, highFreq);
//
//    // preprocessing
//    std::vector<double> inputData = processFileData(mfccComputer, wavPath);
//    std::vector<float> inputDataFloat(inputData.begin(), inputData.end());
//    std::vector<float> temp = inputDataFloat;
//    inputDataFloat.insert(inputDataFloat.end(), temp.begin(), temp.end());
//    inputDataFloat.insert(inputDataFloat.end(), temp.begin(), temp.end());
//
//    // inference
//    std::cout << "WAV input File : " << wavPath << std::endl;
//    std::cout << "ONNX model File : " << modelPath  << std::endl;
//    onnx_inference(inputDataFloat, modelPath);
//    return 0;
//}   



int main() {
    // FIXME:
    const char* wavPath;
    const char* wavPath2;
    const char* modelPath;
    /*const int64_t numChannels = 3;
    const int64_t height = 59;
    const int64_t width = 128;*/
    const int16_t numClasses = 2;

    wavPath = "C:/Users/test/Desktop/Leon/Projects/compute-mfcc/data/1598482996718_21_106.87_108.87_001.wav";
    wavPath2 = "C:/Users/test/Desktop/Leon/Projects/compute-mfcc/data/1606921286802_1_8.93_10.93_001.wav";
    //wavPath = "C:/Users/test/Desktop/Leon/Projects/compute-mfcc/data/1630779176834_42_4.86_6.86_001.wav";
    modelPath = "C:\\Users\\test\\Desktop\\Leon\\Projects\\Snoring_Detection\\checkpoints\\run_050\\snoring.onnx";

    
    loadModelFile model(modelPath, numClasses);

    std::vector<int16_t> waveform = get_data(wavPath);
    std::vector<float> OutputScores = model.run(waveform);

    std::vector<int16_t> waveform2 = get_data(wavPath2);
    std::vector<float> OutputScores2 = model.run(waveform2);
    return 0;
}