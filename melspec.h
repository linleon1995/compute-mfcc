#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <iostream>
#include <fstream>

std::vector<float> preprocess(char* bytearray, int byte_len);