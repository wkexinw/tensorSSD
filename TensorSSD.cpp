#include <cassert>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unistd.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include "TensorSSD.hpp"

using namespace std;

ICudaEngine* TensorSSD::engine;

IRuntime* TensorSSD::runtime;

IExecutionContext* TensorSSD::context;

PluginFactory TensorSSD::pluginFactory;

Profiler TensorSSD::gProfiler;

char* TensorSSD::INPUT_BLOB_NAME = (char*)"data";

char* TensorSSD::OUTPUT_BLOB_NAME = (char*)"detection_out";

int TensorSSD::SOURCE_WIDTH; //width of raw data

int TensorSSD::SOURCE_HEIGHT; //high of raw data

cudaError_t cudaPreImageNetMean(float4* input,
    size_t inputWidth,
    size_t inputHeight,
    float* output,
    size_t outputWidth,
    size_t outputHeight,
    const float3& mean_value);

TensorSSD::TensorSSD()
{
}

TensorSSD::~TensorSSD()
{
}

int TensorSSD::getTensorVersion()
{

  return getInferLibVersion();

}

bool TensorSSD::initialize(int sourceWidth, int sourceHeight)
{

  TensorSSD::engine =
  { nullptr};

  TensorSSD::runtime =
  { nullptr};

  SOURCE_WIDTH = sourceWidth;

  SOURCE_HEIGHT = sourceHeight;

  logger = new Logger("TENSOR_LOGGER");

  f4RGBA = (float4*) malloc(SOURCE_WIDTH*SOURCE_HEIGHT*sizeof(float4));

  return true;

}

bool TensorSSD::modelCacheExists(const char* modelCachePath)
{

  ifstream modelCache(modelCachePath);

  return modelCache ? true : false;

}

float* TensorSSD::allocateMemory(DimsCHW dims, char* info)
{

  float* ptr;

  size_t size;

  size = BATCH_SIZE * dims.c() * dims.h() * dims.w();

#ifdef DEBUG
  printf("[TensorSSD] Allocate memory [%s] size = %d  dims.c() = %d ,dims.h() = %d ,dims.w() = %d\r\n",
      info, (int)size, dims.c(), dims.h(), dims.w());
#endif

  assert(!cudaMallocManaged(&ptr, size * sizeof(float)));

  return ptr;

}

DimsCHW TensorSSD::getTensorDims(const char* name)
{

  for (int b = 0; b < engine->getNbBindings(); b++)
  {

    if (!strcmp(name, engine->getBindingName(b)))

      return static_cast<DimsCHW&&>(engine->getBindingDimensions(b));

  }

  return DimsCHW
  { 0, 0, 0 };

}

bool TensorSSD::imageInference(NvBuffer *buffer, std::vector<cv::Rect>* rectList, int nbBuffer)
{
  int j = 0;

  for (int i = 0; i < SOURCE_WIDTH * SOURCE_HEIGHT; i += 4)
  {

    //printf("%d %d %d %d \n",*(buffer->planes[0].data+i),*(buffer->planes[0].data+i+1),*(buffer->planes[0].data+i+2),*(buffer->planes[0].data+i+3));
    *(f4RGBA + j) = make_float4(*(buffer->planes[0].data + i), *(buffer->planes[0].data + i + 1), *(buffer->planes[0].data + i + 2),
        *(buffer->planes[0].data + i + 3));

    j++;

  }

  memcpy(imgCPU, f4RGBA, SOURCE_WIDTH * sizeof(float4) * SOURCE_HEIGHT);

  if (CUDA_FAILED(cudaPreImageNetMean((float4* )imgCPU, SOURCE_WIDTH, SOURCE_HEIGHT,
      inputData, dimsInputData.w(), dimsInputData.h(), make_float3(127.0f, 127.0f, 127.0f)))
          )
  {

    printf("[TensorSSD] Cuda prepare imagenet mean failed\n");

    return false;
  }

  //assert(engine->getNbBindings() == nbBuffer);

  context->execute(BATCH_SIZE, buffers);

  int i = 0;

  rectList->clear();

  rectList->push_back(cv::Rect(0, 0, 1, 1));

  int tmp = 0;

  for (;;)
  {
    cv::Rect rect(0, 0, 0, 0);

    if ((*(detectOutput + 1 + i * 7) == Bus || *(detectOutput + 1 + i * 7) == Car) && *(detectOutput + 2 + i * 7) > detectConfidence)
    {

//#ifdef DEBUG
//
//      printf("\033[32;43m[TensorSSD] Dection output %d [id = %f , class = %f , confidence = %f , xMin = %f , yMin = %f , xMax = %f , yMax = %f  \033[0m\n",
//          i,
//          *(detectOutput + i * 7),
//          *(detectOutput + 1 + i * 7),
//          *(detectOutput + 2 + i * 7),
//          *(detectOutput + 3 + i * 7),
//          *(detectOutput + 4 + i * 7),
//          *(detectOutput + 5 + i * 7),
//          *(detectOutput + 6 + i * 7));
//#endif

      rect.x = (*(detectOutput + 3 + i * 7) * (float)SOURCE_WIDTH);

      rect.y = (*(detectOutput + 4 + i * 7) * (float)SOURCE_HEIGHT);

      rect.width = (*(detectOutput + 5 + i * 7) * (float)SOURCE_WIDTH) - rect.x;

      rect.height = (*(detectOutput + 6 + i * 7) * (float)SOURCE_HEIGHT) - rect.y;

      rectList->push_back(rect);

      i++;

    }
    else
    {

      i = 0;

      break;

    }

  }

  return true;
}

bool TensorSSD::prepareInference(float confidence)
{

  printf("\033[5;34m ----->>>>> TRT : SOURCE_WIDTH: %d , SOURCE_HEIGHT: %d \033[0m \n", SOURCE_WIDTH, SOURCE_HEIGHT);

  if (!cudaAllocMapped(&imgCPU, &imgCUDA, SOURCE_WIDTH * sizeof(float4) * SOURCE_HEIGHT))
  {

    printf(LOG_CUDA "Cuda failed to allocate (size=%lu)\n", SOURCE_WIDTH * sizeof(float4) * SOURCE_HEIGHT);

    return false;

  }

  detectConfidence = confidence;

  dimsInputData = getTensorDims(INPUT_BLOB_NAME);

  dimsOutputData = getTensorDims(OUTPUT_BLOB_NAME);

  inputData = allocateMemory(dimsInputData, (char*)"input blob");

  detectOutput = allocateMemory(dimsOutputData, (char*)"output blob");

  buffers[0] = inputData;

  buffers[1] = detectOutput;

  context = engine->createExecutionContext();

  context->setProfiler(&gProfiler);

  return true;

}

int TensorSSD::loadTensorSSDModel(const char* tensorSSDModelPath)
{

  stringstream tensorrt_model_stream;

  tensorrt_model_stream.seekg(0, tensorrt_model_stream.beg);

  ifstream tensorrt_model_cache_load(tensorSSDModelPath); //model cache to load

  if (!tensorrt_model_cache_load)
  {

    return 0;

  }

  printf("[TensorSSD] Cached TensorRT model found, start loading...\n");

  tensorrt_model_stream << tensorrt_model_cache_load.rdbuf();

  tensorrt_model_cache_load.close();

  printf("[TensorSSD] Cached TensorRT model load complete.\n");

  // Create inference runtime engine.
  runtime = createInferRuntime(*logger);

  if (!runtime)
  {

    printf("[TensorSSD] Failed to create inference runtime!\n");

    exit (EXIT_FAILURE);

  }

  // support for stringstream deserialization was deprecated in TensorRT v2
  // instead, read the stringstream into a memory buffer and pass that to TRT.
  tensorrt_model_stream.seekg(0, ios::end);

  const int modelSize = tensorrt_model_stream.tellg();

  tensorrt_model_stream.seekg(0, ios::beg);

#ifdef DEBUG

  printf("[TensorSSD] Cached model size : %d\n", modelSize);

#endif

  void* modelMem = malloc(modelSize);

  if (!modelMem)
  {

    printf("[TensorSSD] Failed to allocate memory to deserialize model!\n");

    exit (EXIT_FAILURE);

  }

  tensorrt_model_stream.read((char*)modelMem, modelSize);

  engine = runtime->deserializeCudaEngine(modelMem, modelSize, &pluginFactory);

  free(modelMem);

  if (!engine)
  {

    printf("[TensorSSD] Failed to deserialize CUDA engine!\n");

    exit (EXIT_FAILURE);
  }

#ifdef DEBUG

  printf("[TensorSSD] Deserialize model ok. Number of binding indices %d \n", engine->getNbBindings());

#endif

  tensorrt_model_stream.str("");

  return modelSize;

}

bool TensorSSD::convertCaffeSSDModel(const std::string& deployFile, const std::string& modelFile, const std::string& outFile)
{

  printf("\033[31;32m%s %s\033[0m\n", deployFile.c_str(), modelFile.c_str());

  if (access(deployFile.c_str(), 0) == -1 || access(modelFile.c_str(), 0) == -1)
  {

    return false;

  }
  // create a GIE model from the caffe model and serialize it to a stream
  IHostMemory *gieModelStream
  { nullptr };

  std::vector<std::string> vecOutputBlobs =
  { OUTPUT_BLOB_NAME};

  caffeToGIEModel(deployFile, modelFile, vecOutputBlobs, BATCH_SIZE, &pluginFactory, &gieModelStream);

  // cache the trt model
  std::ofstream trtModelFile(outFile.c_str());

  trtModelFile.write((char *)gieModelStream->data(), gieModelStream->size());

  printf("Convert model to tensor model cache : %s completed.\n", outFile.c_str());

  trtModelFile.close();

  gieModelStream->destroy();

  return true;

}

void TensorSSD::caffeToGIEModel(const std::string& deployFile,             // name for caffe prototxt
    const std::string& modelFile,              // name for model 
    const std::vector<std::string>& outputs,   // network outputs
    unsigned int maxBatchSize,                 // batch size - NB must be at least as large as the batch we want to run with)
    nvcaffeparser1::IPluginFactory* pluginFactory, // factory for plugin layers
    IHostMemory **gieModelStream)              // output stream for the GIE model
{

  // create the builder
  IBuilder* builder = createInferBuilder(*logger);

  // parse the caffe model to populate the network, then set the outputs
  INetworkDefinition* network = builder->createNetwork();

  ICaffeParser* parser = createCaffeParser();

  parser->setPluginFactory(pluginFactory);

  bool fp16 = builder->platformHasFastFp16();

  const IBlobNameToTensor* blobNameToTensor = parser->parse(
      deployFile.c_str(), modelFile.c_str(), *network,
      fp16 ? DataType::kHALF : DataType::kFLOAT);
  // specify which tensors are outputs
  for (auto& s : outputs)
  {
    network->markOutput(*blobNameToTensor->find(s.c_str()));

  }

  // build the engine
  builder->setMaxBatchSize(maxBatchSize);

  builder->setMaxWorkspaceSize(512 << 20);

  /*
   *fp16 unuseable on Tx2 TensorRT(3000) will fix in TensorRT(304) officially
   *more details :https://devtalk.nvidia.com/default/topic/1030035/jetson-tx2/memory-issue-with-half2mode-in-tensorrt-3/
   */
  builder->setHalf2Mode(fp16);

  ICudaEngine* _engine = builder->buildCudaEngine(*network);

  assert(_engine);

  //destroy the network and the parser
  network->destroy();

  parser->destroy();

  // serialize the engine
  (*gieModelStream) = _engine->serialize();

  _engine->destroy();

  builder->destroy();

  shutdownProtobufLibrary();

}

/*
 * Unused
 */
void TensorSSD::printBindings()
{

  for (int bi = 0; bi < engine->getNbBindings(); bi++)
  {

    if (engine->bindingIsInput(bi) == true)
    {

      printf("Binding %d (%s): Input.\n", bi, engine->getBindingName(bi));

    }
    else
    {

      printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));

    }

  }
}

bool TensorSSD::shutdown()
{

  context->destroy();

  engine->destroy();

  runtime->destroy();

  pluginFactory.destroyPlugin();

  return true;

}
