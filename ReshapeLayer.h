#ifndef __RESHAPE_LAYER_H__
#define __RESHAPE_LAYER_H__
#include <cassert>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda_runtime.h> 
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "Common.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

template<int OutC>
class ReshapeLayer: public IPlugin
{
public:
  ReshapeLayer()
  {
  }
  ReshapeLayer(const void* buffer, size_t size)
  {
    assert(size == sizeof(mCopySize));
    mCopySize = *reinterpret_cast<const size_t*>(buffer);
  }

  int getNbOutputs() const override
  {
    return 1;
  }
  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
  {
#ifdef DEBUG
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
#else
    assert(nbInputDims == 1);
    assert(index == 0);
    assert(inputs[index].nbDims == 3);
    assert((inputs[0].d[0]) * (inputs[0].d[1]) % OutC == 0);
    return DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] / OutC, inputs[0].d[2]);
#endif
  }

  int initialize() override
  {
    return 0;
  }

  void terminate() override
  {
  }

  size_t getWorkspaceSize(int) const override
  {
    return 0;
  }

  // currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
  int enqueue(int batchSize, const void* const *inputs, void** outputs, void*, cudaStream_t stream) override
  {
    CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
    return 0;
  }

  size_t getSerializationSize() override
  {
    return sizeof(mCopySize);
  }

  void serialize(void* buffer) override
  {
    *reinterpret_cast<size_t*>(buffer) = mCopySize;
  }

  void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
  {
    mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
  }

protected:
  size_t mCopySize;
};
#endif
