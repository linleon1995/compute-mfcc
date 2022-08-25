#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include "mfcc.cc"
#include "compute-mfcc.cc"


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


int onnx_inference(std::vector<float> inputData) {

    //const int64_t batchSize = 2;
    //bool useCUDA{ true };

    //std::string instanceName{ "image-classification-inference" };
    //// std::string modelFilepath{"../../data/models/squeezenet1.1-7.onnx"};
    //std::string modelFilepath{ "../../data/models/resnet18-v1-7.onnx" };

    //Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
    //    instanceName.c_str());

    //Ort::SessionOptions sessionOptions;
    //sessionOptions.SetIntraOpNumThreads(1);
    //if (useCUDA)
    //{
    //    // Using CUDA backend
    //    // https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L329
    //    OrtCUDAProviderOptions cuda_options{};
    //    sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    //}

    //sessionOptions.SetGraphOptimizationLevel(
    //    GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    //// FIXME:
    //Ort::Session session(env, L"model.onnx", sessionOptions);

    //Ort::AllocatorWithDefaultOptions allocator;

    //size_t numInputNodes = session.GetInputCount();
    //size_t numOutputNodes = session.GetOutputCount();

    //const char* inputName = session.GetInputName(0, allocator);

    //Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    //auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    //ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

    //std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    //if (inputDims.at(0) == -1)
    //{
    //    std::cout << "Got dynamic batch size. Setting input batch size to "
    //        << batchSize << "." << std::endl;
    //    inputDims.at(0) = batchSize;
    //}


    //const char* outputName = session.GetOutputName(0, allocator);

    //Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    //auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    //ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

    //std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    //if (outputDims.at(0) == -1)
    //{
    //    std::cout << "Got dynamic batch size. Setting output batch size to "
    //        << batchSize << "." << std::endl;
    //    outputDims.at(0) = batchSize;
    //}

    //size_t inputTensorSize = vectorProduct(inputDims);
    //std::vector<float> inputTensorValues(inputTensorSize);
    //// Make copies of the same image input.
    ///*for (int64_t i = 0; i < batchSize; ++i)
    //{
    //    std::copy(preprocessedImage.begin<float>(),
    //        preprocessedImage.end<float>(),
    //        inputTensorValues.begin() + i * inputTensorSize / batchSize);
    //}*/

    //size_t outputTensorSize = vectorProduct(outputDims);
    //std::vector<float> outputTensorValues(outputTensorSize);

    //std::vector<const char*> inputNames{ inputName };
    //std::vector<const char*> outputNames{ outputName };
    //std::vector<Ort::Value> inputTensors;
    //std::vector<Ort::Value> outputTensors;

    //Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
    //    OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    //inputTensors.push_back(Ort::Value::CreateTensor<float>(
    //    memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
    //    inputDims.size()));
    //outputTensors.push_back(Ort::Value::CreateTensor<float>(
    //    memoryInfo, outputTensorValues.data(), outputTensorSize,
    //    outputDims.data(), outputDims.size()));

    //session.Run(Ort::RunOptions{ nullptr }, inputNames.data(),
    //    inputTensors.data(), 1 /*Number of inputs*/, outputNames.data(),
    //    outputTensors.data(), 1 /*Number of outputs*/);
    return 0;
}


int main(int argc, char* argv[]) {
    std::string USAGE = "compute-mfcc : MFCC Extractor\n";
    USAGE += "OPTIONS\n";
    USAGE += "--input           : Input 16 bit PCM Wave file\n";
    USAGE += "--output          : Output MFCC file in CSV format, each frame in a line\n";
    USAGE += "--inputlist       : List of input Wave files\n";
    USAGE += "--outputlist      : List of output MFCC CSV files\n";
    USAGE += "--numcepstra      : Number of output cepstra, excluding log-energy (default=12)\n";
    USAGE += "--numfilters      : Number of Mel warped filters in filterbank (default=40)\n";
    USAGE += "--samplingrate    : Sampling rate in Hertz (default=16000)\n";
    USAGE += "--winlength       : Length of analysis window in milliseconds (default=25)\n";
    USAGE += "--frameshift      : Frame shift in milliseconds (default=10)\n";
    USAGE += "--lowfreq         : Filterbank low frequency cutoff in Hertz (default=50)\n";
    USAGE += "--highfreq        : Filterbank high freqency cutoff in Hertz (default=samplingrate/2)\n";
    USAGE += "USAGE EXAMPLES\n";
    USAGE += "compute-mfcc --input input.wav --output output.mfc\n";
    USAGE += "compute-mfcc --input input.wav --output output.mfc --samplingrate 8000\n";
    USAGE += "compute-mfcc --inputlist input.list --outputlist output.list\n";
    USAGE += "compute-mfcc --inputlist input.list --outputlist output.list --numcepstra 17 --samplingrate 44100\n";

    const char* wavPath = getCmdOption(argv, argv + argc, "--input");
    const char* mfcPath = getCmdOption(argv, argv + argc, "--output");
    char* wavListPath = getCmdOption(argv, argv + argc, "--inputlist");
    char* mfcListPath = getCmdOption(argv, argv + argc, "--outputlist");
    char* numCepstraC = getCmdOption(argv, argv + argc, "--numcepstra");
    char* numFiltersC = getCmdOption(argv, argv + argc, "--numfilters");
    char* samplingRateC = getCmdOption(argv, argv + argc, "--samplingrate");
    char* winLengthC = getCmdOption(argv, argv + argc, "--winlength");
    char* frameShiftC = getCmdOption(argv, argv + argc, "--frameshift");
    char* lowFreqC = getCmdOption(argv, argv + argc, "--lowfreq");
    char* highFreqC = getCmdOption(argv, argv + argc, "--highfreq");


    // Check arguments
    if ((argc < 3) || (!(wavPath && mfcPath) && !(wavListPath && mfcListPath))) {
        std::cout << USAGE;
        return 1;
    }

    // Assign variables
    int numCepstra = (numCepstraC ? atoi(numCepstraC) : 12);
    int numFilters = (numFiltersC ? atoi(numFiltersC) : 40);
    int samplingRate = (samplingRateC ? atoi(samplingRateC) : 16000);
    int winLength = (winLengthC ? atoi(winLengthC) : 25);
    int frameShift = (frameShiftC ? atoi(frameShiftC) : 10);
    int lowFreq = (lowFreqC ? atoi(lowFreqC) : 50);
    int highFreq = (highFreqC ? atoi(highFreqC) : samplingRate / 2);

    // Initialise MFCC class instance
    MFCC mfccComputer(samplingRate, numCepstra, winLength, frameShift, numFilters, lowFreq, highFreq);

    // preprocessing
    std::vector<double> inputData = processFileData(mfccComputer, wavPath, mfcPath);
    std::vector<float> inputData_float(inputData.begin(), inputData.end());

    // inference
    onnx_inference(inputData_float);
    return 0;
}   