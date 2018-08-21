#ifndef __FLATTER_LAYER_H__
#define __FLATTER_LAYER_H__
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

class FlattenLayer: public IPlugin
{

protected:

  DimsCHW dimBottom;

  int _size;

public:

  FlattenLayer()
  {
  }

  FlattenLayer(const void* buffer, size_t size)
  {
    assert(size == 3 * sizeof(int));

    const int* d = reinterpret_cast<const int*>(buffer);

    _size = d[0] * d[1] * d[2];

    dimBottom = DimsCHW
        { d[0], d[1], d[2] };

  }

  inline int getNbOutputs() const override
  {
    return 1;
  }
  ;

  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
  {

    assert(1 == nbInputDims);
    assert(0 == index);
    assert(3 == inputs[index].nbDims);

    _size = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];

    return DimsCHW(_size, 1, 1);

  }

  int initialize() override
  {

    return 0;

  }

  inline void terminate() override
  {

  }

  inline size_t getWorkspaceSize(int) const override
  {
    return 0;
  }

  int enqueue(int batchSize, const void* const *inputs, void** outputs, void*, cudaStream_t stream) override
  {

    CHECK(cudaMemcpyAsync(outputs[0], inputs[0], batchSize * _size * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    return 0;

  }

  size_t getSerializationSize() override
  {

    return 3 * sizeof(int);

  }

  void serialize(void* buffer) override
  {

    int* d = reinterpret_cast<int*>(buffer);

    d[0] = dimBottom.c();
    d[1] = dimBottom.h();
    d[2] = dimBottom.w();

  }

  void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
  {

    dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);

  }

};

#endif
