// -----------------------------------------------------------------------------
// Wrapper for MFCC feature extractor
// -----------------------------------------------------------------------------
//
//  Copyright (C) 2016 D S Pavan Kumar
//  dspavankumar [at] gmail [dot] com

#include <algorithm>
#include <iostream>
#include <fstream>
#include "mfcc.cc"

// A simple option parser
char* getCmdOption(char **begin, char **end, const std::string &value) {
    char **iter = std::find(begin, end, value);
    if (iter != end && ++iter != end)
        return *iter;
    return nullptr;
}

// Process each file
int processFile (MFCC &mfccComputer, const char* wavPath, const char* mfcPath) {
    // Initialise input and output streams    
    std::ifstream wavFp;
    std::ofstream mfcFp;
    
    // Check if input is readable
    wavFp.open(wavPath, std::ios::binary);
    if (!wavFp.is_open()) {
        std::cerr << "Unable to open input file: " << wavPath << std::endl;
        return 1;
    }
    
    // Check if output is writable
    mfcFp.open(mfcPath, std::ios::binary);
    if (!mfcFp.is_open()) {
        std::cerr << "Unable to open output file: " << mfcPath << std::endl;
        wavFp.close();
        return 1;
    }
   
    // Extract and write features
    if (mfccComputer.process (wavFp, mfcFp))
        std::cerr << "Error processing " << wavPath << std::endl;

    wavFp.close();
    mfcFp.close();
    return 0;
}


// Process each file
std::vector <double> processFileData(MFCC& mfccComputer, const char* wavPath, const char* mfcPath) {
    // Initialise input and output streams    
    std::ifstream wavFp;
    std::ofstream mfcFp;

    //// Check if input is readable
    //wavFp.open(wavPath, std::ios::binary);
    //if (!wavFp.is_open()) {
    //    std::cerr << "Unable to open input file: " << wavPath << std::endl;
    //    return 1;
    //}

    //// Check if output is writable
    //mfcFp.open(mfcPath, std::ios::binary);
    //if (!mfcFp.is_open()) {
    //    std::cerr << "Unable to open output file: " << mfcPath << std::endl;
    //    wavFp.close();
    //    return 1;
    //}

    //// Extract and write features
    //if (mfccComputer.process_and_return(wavFp, mfcFp))
    //    std::cerr << "Error processing " << wavPath << std::endl;

    std::vector<double> inputData = mfccComputer.process_and_return(wavFp, mfcFp);

    wavFp.close();
    mfcFp.close();
    return inputData;
}

// Process lists
int processList (MFCC &mfccComputer, const char* wavListPath, const char* mfcListPath) {
    std::ifstream wavListFp, mfcListFp;

    // Check if wav list is readable
    wavListFp.open(wavListPath);
    if (!wavListFp.is_open()) {
        std::cerr << "Unable to open input list: " << wavListPath << std::endl;
        return 1;
    }

    // Check if mfc list is readable
    mfcListFp.open(mfcListPath);
    if (!mfcListFp.is_open()) {
        std::cerr << "Unable to open output list: " << mfcListPath << std::endl;
        return 1;
    }

    // Process lists
    std::string wavPath, mfcPath;
    while (true) {
        std::getline (wavListFp, wavPath);
        std::getline (mfcListFp, mfcPath);
        if (wavPath.empty() || mfcPath.empty()) {
            wavListFp.close();
            mfcListFp.close();
            return 0;
        }
        if (processFile (mfccComputer, wavPath.c_str(), mfcPath.c_str())) {
            wavListFp.close();
            mfcListFp.close();
            return 1;
        }
    }
}


// find the file size
int getFileSize(FILE* inFile)
{
    int fileSize = 0;
    fseek(inFile, 0, SEEK_END);

    fileSize = ftell(inFile);

    fseek(inFile, 0, SEEK_SET);
    return fileSize;
}


// Main
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

    const char *wavPath = getCmdOption(argv, argv+argc, "--input");
    const char *mfcPath = getCmdOption(argv, argv+argc, "--output");
    char *wavListPath = getCmdOption(argv, argv+argc, "--inputlist");
    char *mfcListPath = getCmdOption(argv, argv+argc, "--outputlist");
    char *numCepstraC = getCmdOption(argv, argv+argc, "--numcepstra");
    char *numFiltersC = getCmdOption(argv, argv+argc, "--numfilters");
    char *samplingRateC = getCmdOption(argv, argv+argc, "--samplingrate");
    char *winLengthC = getCmdOption(argv, argv+argc, "--winlength");
    char *frameShiftC = getCmdOption(argv, argv+argc, "--frameshift");
    char *lowFreqC = getCmdOption(argv, argv+argc, "--lowfreq");
    char *highFreqC = getCmdOption(argv, argv+argc, "--highfreq");


    // Check arguments
    if ((argc<3) || (!(wavPath && mfcPath) && !(wavListPath && mfcListPath))) {
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
    int highFreq = (highFreqC ? atoi(highFreqC) : samplingRate/2);

    // Initialise MFCC class instance
    MFCC mfccComputer (samplingRate, numCepstra, winLength, frameShift, numFilters, lowFreq, highFreq);

    // Process wav files
    if (wavPath && mfcPath)
        if (processFile (mfccComputer, wavPath, mfcPath))
            return 1;
            
    // Process lists
    if (wavListPath && mfcListPath)
        if (processList (mfccComputer, wavListPath, mfcListPath))
            return 1;

    return 0;
}



