#include <algorithm>
#include <iostream>
#include <fstream>
#include<numeric>
#include<complex>
#include<vector>
#include<map>
#include<math.h>


std::vector<float> preprocess(char* bytearray, int byte_len);
std::vector<std::vector<std::vector<std::vector<float>>>> preprocess_2d(char* bytearray, int byte_len);


class MelSpec {
public:
    MelSpec(int sampFreq = 16000, int nCep = 12, int winLength = 25, int frameShift = 10, int numFilt = 40, double lf = 50, double hf = 8000);

private:
    typedef std::vector<float> v_f;
    typedef std::vector<double> v_d;
    typedef std::complex<double> c_d;
    typedef std::vector<v_d> m_d;
    typedef std::vector<c_d> v_c_d;
    typedef std::map<int, std::map<int, c_d> > twmap;

public:
    int16_t height = 0; //spectrogram height
    int16_t width = 0; //spectrogram width
    int16_t get_height() const { return height; }

    std::string processFrame(int16_t* samples, size_t N);
    v_d processFrame_vect(int16_t* samples, size_t N);
    v_d WavformtoMelspec(std::vector<int16_t> waveform);
    v_d processBytes(char* bytearray, int byte_len);
    std::vector<float> preprocess(char* bytearray, int byte_len);

    // 2d
    std::vector<v_f> WavformtoMelspec_2d(std::vector<int16_t> waveform);
    std::vector<v_f> processBytes_2d(char* bytearray, int byte_len);
    std::vector<std::vector<v_f>> preprocess_2d(char* bytearray, int byte_len);


private:
    const double PI = 4*atan(1.0);   // Pi = 3.14...
    int fs;

    twmap twiddle;
    size_t winLengthSamples, frameShiftSamples, numCepstra, numFFT, numFFTBins, numFilters;
    double preEmphCoef, lowFreq, highFreq;
    v_d frame, powerSpectralCoef, lmfbCoef, hamming, mfcc, prevsamples;
    m_d fbank, dct;

    inline double hz2mel(double f);
    inline double mel2hz(double m);
    void compTwiddle(void);
    v_c_d fft(v_c_d x);
    void preEmphHam(void);
    void computePowerSpec(void);
    void applyLMFB(void);
    void applyDct(void);
    void initHamDct(void);
    void initFilterbank();
    std::string v_d_to_string(v_d vec);
};