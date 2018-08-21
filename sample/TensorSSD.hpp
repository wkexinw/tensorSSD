#ifndef __TENSOR_SSD_H__
#define __TENSOR_SSD_H__

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvCaffeParser.h>
#include "PluginFactory.h"

#include "Logger.h"

#define OUTPUT_CLS_SIZE 21 //SSD default

#define MODEL_INPUT_C 3 //SSD default
#define MODEL_INPUT_H 300//SSD default
#define MODEL_INPUT_W 300//SSD default
#define BATCH_SIZE  1

#define DEBUG 1

/* 
 * Unused
 * Not enough training except Car=7
 */
enum DETECT_CATEGORY
{
  Background = 0, Aeroplane, Bicyle,
  Bird, Boat, Bottle, Bus = 6, Car = 7, Cat, Chair, Cow,
  Diningtable, Dog, Horse, Motorbike, Person,
  Pottedplant, Sheep, Sofa, Train, Tvmonitor
};

class TensorSSD
{
private:
  Logger* logger;

  static ICudaEngine* engine;

  static IRuntime* runtime;

  static IExecutionContext* context;

  static PluginFactory pluginFactory;

  static Profiler gProfiler;

  static int SOURCE_WIDTH;

  static int SOURCE_HEIGHT;

  static char* INPUT_BLOB_NAME;

  static char* OUTPUT_BLOB_NAME;

  void* buffers[2];

  float* inputData;

  DimsCHW dimsInputData;

  DimsCHW dimsOutputData;

  float detectConfidence;

  float* detectOutput;

public:

  TensorSSD();

  ~TensorSSD();

  int getTensorVersion(); //get tensor version

  bool initialize(int sourceWidth, int sourceHeight); //initialize

  bool shutdown(); //shutdown

  bool modelCacheExists(const char* modelCachePath);

  int loadTensorSSDModel(const char* tensorSSDModelPath); //return model size

  bool convertCaffeSSDModel(const std::string& deployFile,
      const std::string& modelFile,
      const std::string& outFile);

  void prepareInference(float confidence); //prepare

  int imageInference(float4* imgRGBA, std::vector<cv::Rect>* rectList, int nbBuffer = 2); //nbBuffer =2 (input & output)

private:

  void caffeToGIEModel(const std::string& deployFile,             // name for caffe prototxt
      const std::string& modelFile,              // name for model 
      const std::vector<std::string>& outputs,   // network outputs
      unsigned int maxBatchSize,                 // batch size - NB must be at least as large as the batch we want to run with)
      nvcaffeparser1::IPluginFactory* pluginFactory, // factory for plugin layers
      IHostMemory **gieModelStream);              // output stream for the GIE model

  DimsCHW getTensorDims(const char* name);

  float* allocateMemory(DimsCHW dims, char* info);

  void printBindings();

};

#endif
