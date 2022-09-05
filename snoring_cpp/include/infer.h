#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <iostream>
#include <fstream>


class loadModelFile {
    public:
        loadModelFile(const char* modelPath);

        std::vector<float> run(char* bytearray, int byte_len);

        std::tuple<std::vector<float>, std::vector<float>> run_and_get_input(char* bytearray, int byte_len);

        std::vector<float> preprocess(char* bytearray, int byte_len);

    private:
        int32_t height;
        int32_t width;
        int32_t numInputElements;
        int32_t numChannels = 3;
        int32_t numClasses = 2;

        char* input_names[1]{ nullptr };
        char* output_names[1]{ nullptr };

        std::array<int64_t, 4> inputShape = {1, numChannels, -1, -1};
        std::array<int64_t, 2> outputShape = { 1, numClasses };

        Ort::Value inputTensor{ (Ort::Value)nullptr };
        Ort::Value outputTensor{ (Ort::Value)nullptr };

        Ort::Env env;
        Ort::RunOptions runOptions;
        Ort::Session session_{ nullptr };

        std::array<float, 2> results;

};