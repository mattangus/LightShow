#include <memory>
#include <string>
#include <iostream>
#include <unordered_map>
#include <random>

#include <aquila/aquila.h>
#include <aquila/source/WaveFile.h>
#include <aquila/transform/Mfcc.h>

#include <aquila/transform/OouraFft.h>

#include <opencv2/opencv.hpp>
extern "C" {
    #include "mailbox.h"
    #include "gpu_fft.h"
}

#include <unistd.h>

#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/optional_debug_tools.h>

#include <cmath>
#include <iterator>
#include <functional>
#include <numeric>
#include <type_traits>
#include <chrono>

#include <omp.h>

const double pi = std::acos(-1);

template <typename It>
void softmax (It beg, It end)
{
  using VType = typename std::iterator_traits<It>::value_type;

  static_assert(std::is_floating_point<VType>::value,
                "Softmax function only applicable for floating types");

  auto max_ele { *std::max_element(beg, end) };

  std::transform(
      beg,
      end,
      beg,
      [&](VType x){ return std::exp(x - max_ele); });

  VType exptot = std::accumulate(beg, end, 0.0);

  std::transform(
      beg,
      end,
      beg,
      std::bind2nd(std::divides<VType>(), exptot));  
}

/*
need custom reader since aquila doesn't take into account extra headers.
*/
class CustomReader
{
private:
public:
    struct WaveHeader
    {
        char   RIFF[4];
        std::uint32_t DataLength;
        char   WAVE[4];
    };
    struct ChunkHeader
    {   
        char   id[4];
        std::uint32_t length;
    };
    struct FmtData
    {
        std::uint16_t formatTag;
        std::uint16_t Channels;
        std::uint32_t SampFreq;
        std::uint32_t BytesPerSec;
        std::uint16_t BytesPerSamp;
        std::uint16_t BitsPerSamp;
    };
    WaveHeader header;
    FmtData fmt;
    uint32_t waveSize;

    Aquila::WaveFile::ChannelType leftChannel;
    Aquila::WaveFile::ChannelType rightChannel;

    std::string filename;

    CustomReader(std::string filename) : filename(filename)
    {

    }
    ~CustomReader()
    {

    }

    void readHeaderAndChannels()
    {
        std::fstream fs;
        fs.open(filename.c_str(), std::ios::in | std::ios::binary);
        fs.read((char*)(&header), sizeof(WaveHeader));
        if(std::string(header.RIFF,4) != "RIFF")
        {
            std::cerr << "not an RIFF" << std::endl;
            fs.close();
            exit(1);
        }
        if(std::string(header.WAVE,4) != "WAVE")
        {
            std::cerr << "not a WAVE" << std::endl;
            fs.close();
            exit(1);
        }
        short* data = nullptr;
        ChunkHeader curChunk;
        while(!fs.eof())
        {
            // std::cout << "reading chunk header" << std::endl;
            fs.read((char*)(&curChunk), sizeof(ChunkHeader));
            if(std::string(curChunk.id,4) == "fmt ")
            {
                fs.read((char*)(&fmt), sizeof(FmtData));
                // std::cout << "found fmt." << std::endl;
                // std::cout << "formatTag: " << fmt.formatTag << std::endl;
                // std::cout << "Channels: " << fmt.Channels << std::endl;
                // std::cout << "SampFreq: " << fmt.SampFreq << std::endl;
                // std::cout << "BytesPerSec: " << fmt.BytesPerSec << std::endl;
                // std::cout << "BytesPerSamp: " << fmt.BytesPerSamp << std::endl;
                // std::cout << "BitsPerSamp: " << fmt.BitsPerSamp << std::endl;
            }
            else if(std::string(curChunk.id,4) == "data")
            {
                data = new short[curChunk.length/2];
                waveSize = curChunk.length;
                // std::cout << "found data. Length: " << waveSize << std::endl;
                fs.read((char*)data, curChunk.length);
                break;
            }
            else
            {
                // std::cout << "found other chunk: '" << std::string(curChunk.id,4)
                //         << "'. Length: " << curChunk.length << std::endl;
                fs.ignore(curChunk.length);
            }
        }

        fs.close();

        if(data == nullptr)
        {
            // std::cerr << "no data chunk found" << std::endl;
            exit(1);
        }

        Aquila::WaveFileHandler fh("notused");

        // initialize data channels (using right channel only in stereo mode)
        unsigned int channelSize = waveSize/fmt.BytesPerSamp;
        leftChannel.resize(channelSize);
        if (2 == fmt.Channels)
            rightChannel.resize(channelSize);

        // most important conversion happens right here
        if (16 == fmt.BitsPerSamp)
        {
            if (2 == fmt.Channels)
            {
                // std::cout << "decode16bitStereo" << std::endl;
                fh.decode16bitStereo(leftChannel, rightChannel, data, channelSize);
            }
            else
            {
                // std::cout << "decode16bit" << std::endl;
                fh.decode16bit(leftChannel, data, channelSize);
            }
        }
        else
        {
            if (2 == fmt.Channels)
            {
                // std::cout << "decode8bitStereo" << std::endl;
                fh.decode8bitStereo(leftChannel, rightChannel, data, channelSize);
            }
            else
            {
                // std::cout << "decode8bit" << std::endl;
                fh.decode8bit(leftChannel, data, channelSize);
            }
        }

        // clear the buffer
        delete [] data;
    }

    /**
     * Returns the audio recording length
     *
     * @return recording length in milliseconds
     */
    unsigned int getAudioLength() const
    {
        return static_cast<unsigned int>(waveSize /
                static_cast<double>(fmt.BytesPerSec) * 1000);
    }
};


std::vector<std::complex<double>> fftw(std::vector<std::complex<double>>& in)
{
    
	int mbox = mbox_open();
    int len = in.size();
	int logLen = 11; //TODO: unhardcode
    if((1 << logLen) != len)
    {
        std::cout << "log is wrong" << std::endl;
        mbox_close(mbox);
        exit(0);
    }
	int direction = GPU_FFT_FWD;
	int jobs = 1;

	struct GPU_FFT *fft;
	int ret = gpu_fft_prepare(mbox, logLen, direction, jobs, &fft);
    switch(ret) {
        case -1: printf("Unable to enable V3D. Please check your firmware is up to date.\n"); exit(ret);
        case -2: printf("log2_N=%d not supported.  Try between 8 and 22.\n", logLen);         exit(ret);
        case -3: printf("Out of memory.  Try a smaller batch or increase GPU memory.\n");     exit(ret);
        case -4: printf("Unable to map Videocore peripherals into ARM memory space.\n");      exit(ret);
        case -5: printf("Can't open libbcm_host.\n");                                         exit(ret);
    }

    struct GPU_FFT_COMPLEX *fft_in = fft->in;

    /*setting all mem to 0 in order to have guard space between buffers*/
    for (int k=0; k < len; k++)
    {
        fft_in[k].re=fft_in[k].im=0; 
    }

    for(int i = 0; i < in.size(); i++)
    {
        fft_in[i].re = (float)in[i].real();
        fft_in[i].im = (float)in[i].imag();
    }

    usleep(1);
    gpu_fft_execute(fft);

    /*assigning buffer address to pointer out*/
    struct GPU_FFT_COMPLEX *fft_out = fft->out; 

    std::vector<std::complex<double>> out(in.size());
    for(int i = 0; i < out.size(); i++)
    {
        out[i] = {(double)fft_out[i].re, (double)fft_out[i].im};
    }

	gpu_fft_release(fft);

    return out;
}

std::vector<std::vector<double>> spectrogram(std::vector<double> wave,
                                            int sampFreq,
                                            int bankSize,
                                            int window_size = 2048,
                                            int hop_length = 512)
{   
    std::vector<std::vector<double>> spect;
    std::vector<double> multipliers(window_size);
    //#pragma omp parallel for
    for(int i = 0; i < window_size; i++)
    {
        multipliers[i] = 0.5 * (1 - std::cos(2*pi*i/(window_size - 1)));
    }

    Aquila::SignalSource source(wave, sampFreq);
    Aquila::MelFilterBank bank(sampFreq, window_size, bankSize, 128);

    std::vector<std::complex<double>> complexWave(wave.size());
    //#pragma omp parallel for
    for (std::size_t i = 0; i <  wave.size(); ++i)
    {
        complexWave[i] = {wave[i], 0};
    }

    //#pragma omp parallel for
    for(int start = 0; start < wave.size() - window_size; start += hop_length + window_size)
    {
        std::vector<std::complex<double>> vals(window_size);
        for(int i = 0; i < window_size; i++)
        {
            vals[i] = complexWave[start + i]*multipliers[i];
        }
        auto tform = fftw(vals);
        auto filterOutput = bank.applyAll(tform);
        spect.push_back(filterOutput);
    }

    // Aquila::FramesCollection frames(source, window_size, hop_length);
    // std::cout << "num frames: " << frames.count() << std::endl;
    // //Aquila::Mfcc mfcc(window_size);

    // auto m_fft = Aquila::FftFactory::getFft(window_size);

    // #pragma omp parallel for
    // for (int i = 0; i < spect.size(); i++) {
    //     const Aquila::Frame* frame = &(frames.begin()[i]);
    //     auto farray = frame->toArray();
    //     std::vector<double> vals(window_size);
    //     //#pragma omp parallel for
    //     for(int i = 0; i < window_size; i++)
    //     {
    //         vals[i] = farray[i]*multipliers[i];
    //     }
    //     auto spectrum = m_fft->fft(&vals[0]);
    //     auto filterOutput = bank.applyAll(spectrum);

    //     //auto mfccValues = mfcc.calculate(frame, 128);
    //     spect[i] = filterOutput;
    // }

    // for(int start = 0; start < wave.size() - window_size; start += hop_length)
    // {
    //     std::vector<double> window(window_size);
    //     for(int i = 0; i < window_size; i++)
    //     {
    //         double multiplier = 0.5 * (1 - std::cos(2*pi*i/(window_size - 1)));
    //         window[i] = multiplier * wave[i + start];
    //     }

    //     std::vector<std::complex<double>> tform = fft(window);

    //     for(int i = 0; i < window_size; i++)
    //     {
    //         window[i] = std::norm(tform[i]);
    //     }

    //     spect.push_back(window);
    // }

    return spect;

}

std::vector<double> getAudio(std::string filename, int* sampFreq)
{
    CustomReader reader(filename);
    reader.readHeaderAndChannels();


    // std::cout << "wavesize: " << reader.waveSize << std::endl;
    int m_inputSize = reader.getAudioLength();
    std::cout << "input size: " << m_inputSize << std::endl;
    std::cout << "left channel size: " << reader.leftChannel.size() << std::endl;
    
    *sampFreq = reader.fmt.SampFreq;
    return reader.leftChannel;
}

void printTensorInfo(TfLiteTensor* t)
{
    std::cout << "name: " << t->name << std::endl;
    const char* s = 0;
#define PROCESS_VAL(p) case(p): s = #p; break;
    switch(t->type){
        PROCESS_VAL(TfLiteType::kTfLiteNoType);     
        PROCESS_VAL(TfLiteType::kTfLiteFloat32);     
        PROCESS_VAL(TfLiteType::kTfLiteInt32);
        PROCESS_VAL(TfLiteType::kTfLiteUInt8);
        PROCESS_VAL(TfLiteType::kTfLiteInt64);
        PROCESS_VAL(TfLiteType::kTfLiteString);
        PROCESS_VAL(TfLiteType::kTfLiteBool);
        PROCESS_VAL(TfLiteType::kTfLiteInt16);
        PROCESS_VAL(TfLiteType::kTfLiteComplex64);
        PROCESS_VAL(TfLiteType::kTfLiteInt8);
        default:
            s = "no type found";
            break;
    }
#undef PROCESS_VAL
    std::cout << "type: " << s << std::endl;

    std::cout << "dims: (";
    for(int i = 0; i < t->dims->size; i++)
    {
        std::cout << t->dims->data[i] << ", ";
    }
    std::cout << ")" << std::endl;
}

std::string type2str(int type) {
  std::string r;

  uint8_t depth = type & CV_MAT_DEPTH_MASK;
  uint8_t chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

cv::Mat padImg(cv::Mat& img, int height, int width)
{
    cv::Mat padded;
    int padding = 3;
    padded.create(img.rows + 2*padding, img.cols + 2*padding, img.type());
    padded.setTo(cv::Scalar::all(0));

    img.copyTo(padded(cv::Rect(padding, padding, img.cols, img.rows)));

    return padded;
}

float RandomFloat(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

template<typename _Tp>
cv::Mat toMat(const std::vector<std::vector<_Tp> > vecIn) {
    cv::Mat_<_Tp> matOut(vecIn.size(), vecIn.at(0).size());
    for (int i = 0; i < matOut.rows; ++i) {
        for (int j = 0; j < matOut.cols; ++j) {
            matOut(i, j) = vecIn.at(i).at(j);
        }
    }
    return matOut;
}

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

int main(int argc, char *argv[])
{
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("frozen_float.tflite");
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    //Resize input tensors, if desired.
    interpreter->AllocateTensors();

    auto* input_tensor = interpreter->tensor(interpreter->inputs()[0]);
    auto* output_tensor = interpreter->tensor(interpreter->outputs()[0]);
    std::cout << "input:" << std::endl;
    printTensorInfo(input_tensor);
    std::cout << "outputs:" << std::endl;
    printTensorInfo(output_tensor);

    float* input = interpreter->typed_input_tensor<float>(0);
    // Fill `input`.

    int sampFreq;
    std::vector<std::vector<double>> spect;
    std::vector<double> wave = getAudio(argv[1], &sampFreq);
    float totalMs = 0;
    int n = 3;
    for(int i = 0; i < n; i++)
    {
        auto t0 = Time::now();
        spect = spectrogram(wave, sampFreq, atoi(argv[2]));
        auto t1 = Time::now();
        fsec fs = t1 - t0;
        std::cout << fs.count() << std::endl;
        if(i != 0)
            totalMs += fs.count();
    }
    std::cout << "spect sec: " << (totalMs/(n-1)) << std::endl;
    std::cout << "spect shape: " << spect.size() << ", " << spect[0].size() << std::endl;

    cv::Mat img = toMat(spect);
    cv::transpose(img, img);

    // cv::Mat img = cv::imread(argv[1], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    std::cout << img.rows << ", " << img.cols << std::endl;
    //img -= cv::mean(img);
    // cv::resize(img, img, cv::Size(150, 128));
    std::cout << type2str(img.type()) << std::endl;
    std::cout << img.rows << ", " << img.cols << std::endl;
    float* img_data = img.ptr<float>();

    double min, max;
    cv::minMaxLoc(img, &min, &max);
    std::cout << "min: " << min << " max: " << max << std::endl;

    cv::Mat logimg;
    cv::log(img + 0.00001, logimg);
    cv::minMaxLoc(logimg, &min, &max);
    logimg -= min;
    logimg /= max;
    logimg *= 255;
    logimg.convertTo(logimg, CV_8U);

    cv::imwrite("test.png", logimg);

    // std::copy(img_data, img_data + (img.rows*img.cols), input);

    totalMs = 0;
    n = 31;
    for(int i = 0; i < n; i++)
    {
        auto t0 = Time::now();
        interpreter->Invoke();
        auto t1 = Time::now();
        fsec fs = t1 - t0;
        if(i != 0)
            totalMs += fs.count();
    }

    std::cout << "avg sec: " << (totalMs/(n-1)) << std::endl;

    std::cout << "done running" << std::endl;
    float* output = interpreter->typed_output_tensor<float>(0);

    softmax(output, output + 7);

    std::vector<std::string> id_to_name = {"angry", "disgust", "fear", "happy", "neutral", "surprise", "sad"};

    int max_i = 0;

    std::cout << "prediction: (";
    for(int i = 0; i < 7; i++)
    {
        if(output[i] > output[max_i])
        {
            max_i = i;
        }
        std::cout << output[i] << ", ";
    }
    std::cout << ")" << std::endl;
    std::cout << "prediction: " << id_to_name[max_i] << std::endl;

    return 0;
}