#ifndef __PLUGIN_FACTORY_H__
#define __PLUGIN_FACTORY_H__

#include <algorithm>
#include <cassert>
#include <iostream>
#include <cstring>
#include <sys/stat.h>
#include <map>
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "FlattenLayer.h"
#include "ReshapeLayer.h"

using namespace nvinfer1;

using namespace nvcaffeparser1;

using namespace plugin;

struct Profiler: public IProfiler
{

  typedef std::pair<std::string, float> Record;

  std::vector<Record> mProfile;

  virtual void reportLayerTime(const char* layerName, float ms)
  {
    auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r)
    { return r.first == layerName;});

    if (record == mProfile.end())
    {

      mProfile.push_back(std::make_pair(layerName, ms));

    }
    else
    {

      record->second += ms;

    }

  }

  void printLayerTimes(const int TIMING_ITERATIONS)
  {

    float totalTime = 0;

    for (size_t i = 0; i < mProfile.size(); i++)
    {

      printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);

      totalTime += mProfile[i].second;

    }

    printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);

  }

};

class PluginFactory: public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{

public:

  // caffe parser plugin implementation
  bool isPlugin(const char* name) override
  {

    return (!strcmp(name, "conv4_3_norm")
        || !strcmp(name, "conv4_3_norm_mbox_conf_perm")
        || !strcmp(name, "conv4_3_norm_mbox_conf_flat")
        || !strcmp(name, "conv4_3_norm_mbox_loc_perm")
        || !strcmp(name, "conv4_3_norm_mbox_loc_flat")
        || !strcmp(name, "fc7_mbox_conf_perm")
        || !strcmp(name, "fc7_mbox_conf_flat")
        || !strcmp(name, "fc7_mbox_loc_perm")
        || !strcmp(name, "fc7_mbox_loc_flat")
        || !strcmp(name, "conv6_2_mbox_conf_perm")
        || !strcmp(name, "conv6_2_mbox_conf_flat")
        || !strcmp(name, "conv6_2_mbox_loc_perm")
        || !strcmp(name, "conv6_2_mbox_loc_flat")
        || !strcmp(name, "conv7_2_mbox_conf_perm")
        || !strcmp(name, "conv7_2_mbox_conf_flat")
        || !strcmp(name, "conv7_2_mbox_loc_perm")
        || !strcmp(name, "conv7_2_mbox_loc_flat")
        || !strcmp(name, "conv8_2_mbox_conf_perm")
        || !strcmp(name, "conv8_2_mbox_conf_flat")
        || !strcmp(name, "conv8_2_mbox_loc_perm")
        || !strcmp(name, "conv8_2_mbox_loc_flat")
        || !strcmp(name, "conv9_2_mbox_conf_perm")
        || !strcmp(name, "conv9_2_mbox_conf_flat")
        || !strcmp(name, "conv9_2_mbox_loc_perm")
        || !strcmp(name, "conv9_2_mbox_loc_flat")
        || !strcmp(name, "conv4_3_norm_mbox_priorbox")
        || !strcmp(name, "fc7_mbox_priorbox")
        || !strcmp(name, "conv6_2_mbox_priorbox")
        || !strcmp(name, "conv7_2_mbox_priorbox")
        || !strcmp(name, "conv8_2_mbox_priorbox")
        || !strcmp(name, "conv9_2_mbox_priorbox")
        || !strcmp(name, "mbox_conf_reshape")
        || !strcmp(name, "mbox_conf_flatten")
        || !strcmp(name, "mbox_loc")
        || !strcmp(name, "mbox_conf")
        || !strcmp(name, "mbox_priorbox")
        || !strcmp(name, "detection_out")
        || !strcmp(name, "detection_out2")
    );

  }

  virtual IPlugin* createPlugin(const char* layerName, const Weights* weights, int nbWeights) override
  {
    // there's no way to pass parameters through from the model definition, so we have to define it here explicitly
    if (!strcmp(layerName, "conv4_3_norm"))
    {

      _nvPlugins[layerName] = plugin::createSSDNormalizePlugin(weights, false, false, 1e-10);

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "conv4_3_norm_mbox_loc_perm")
        || !strcmp(layerName, "conv4_3_norm_mbox_conf_perm")
        || !strcmp(layerName, "fc7_mbox_loc_perm")
        || !strcmp(layerName, "fc7_mbox_conf_perm")
        || !strcmp(layerName, "conv6_2_mbox_loc_perm")
        || !strcmp(layerName, "conv6_2_mbox_conf_perm")
        || !strcmp(layerName, "conv7_2_mbox_loc_perm")
        || !strcmp(layerName, "conv7_2_mbox_conf_perm")
        || !strcmp(layerName, "conv8_2_mbox_loc_perm")
        || !strcmp(layerName, "conv8_2_mbox_conf_perm")
        || !strcmp(layerName, "conv9_2_mbox_loc_perm")
        || !strcmp(layerName, "conv9_2_mbox_conf_perm")
            )
    {

      _nvPlugins[layerName] = plugin::createSSDPermutePlugin(Quadruple(
          { 0, 2, 3, 1 }));

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "conv4_3_norm_mbox_priorbox"))
    {

      plugin::PriorBoxParameters params =
          { 0 };
      float minSize[1] =
          { 30.0f };
      float maxSize[1] =
          { 60.0f };
      float aspectRatios[1] =
          { 2.0f };
      params.minSize = (float*)minSize;
      params.maxSize = (float*)maxSize;
      params.aspectRatios = (float*)aspectRatios;
      params.numMinSize = 1;
      params.numMaxSize = 1;
      params.numAspectRatios = 1;
      params.flip = true;
      params.clip = false;
      params.variance[0] = 0.10000000149;
      params.variance[1] = 0.10000000149;
      params.variance[2] = 0.20000000298;
      params.variance[3] = 0.20000000298;
      params.imgH = 0;
      params.imgW = 0;
      params.stepH = 8.0f;
      params.stepW = 8.0f;
      params.offset = 0.5f;

      _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "fc7_mbox_priorbox"))
    {

      plugin::PriorBoxParameters params =
          { 0 };
      float minSize[1] =
          { 60.0f };
      float maxSize[1] =
          { 111.0f };
      float aspectRatios[2] =
          { 2.0f, 3.0f };
      params.minSize = (float*)minSize;
      params.maxSize = (float*)maxSize;
      params.aspectRatios = (float*)aspectRatios;
      params.numMinSize = 1;
      params.numMaxSize = 1;
      params.numAspectRatios = 2;
      params.flip = true;
      params.clip = false;
      params.variance[0] = 0.10000000149;
      params.variance[1] = 0.10000000149;
      params.variance[2] = 0.20000000298;
      params.variance[3] = 0.20000000298;
      params.imgH = 0;
      params.imgW = 0;
      params.stepH = 16.0f;
      params.stepW = 16.0f;
      params.offset = 0.5f;

      _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "conv6_2_mbox_priorbox"))
    {

      plugin::PriorBoxParameters params =
          { 0 };
      float minSize[1] =
          { 111.0f };
      float maxSize[1] =
          { 162.0f };
      float aspectRatios[2] =
          { 2.0f, 3.0f };
      params.minSize = (float*)minSize;
      params.maxSize = (float*)maxSize;
      params.aspectRatios = (float*)aspectRatios;
      params.numMinSize = 1;
      params.numMaxSize = 1;
      params.numAspectRatios = 2;
      params.flip = true;
      params.clip = false;
      params.variance[0] = 0.10000000149;
      params.variance[1] = 0.10000000149;
      params.variance[2] = 0.20000000298;
      params.variance[3] = 0.20000000298;
      params.imgH = 0;
      params.imgW = 0;
      params.stepH = 32.0f;
      params.stepW = 32.0f;
      params.offset = 5.0f;

      _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "conv7_2_mbox_priorbox"))
    {

      plugin::PriorBoxParameters params =
          { 0 };
      float minSize[1] =
          { 162.0f };
      float maxSize[1] =
          { 213.0f };
      float aspectRatios[2] =
          { 2.0f, 3.0f };
      params.minSize = (float*)minSize;
      params.maxSize = (float*)maxSize;
      params.aspectRatios = (float*)aspectRatios;
      params.numMinSize = 1;
      params.numMaxSize = 1;
      params.numAspectRatios = 2;
      params.flip = true;
      params.clip = false;
      params.variance[0] = 0.10000000149;
      params.variance[1] = 0.10000000149;
      params.variance[2] = 0.20000000298;
      params.variance[3] = 0.20000000298;
      params.imgH = 0;
      params.imgW = 0;
      params.stepH = 64.0f;
      params.stepW = 64.0f;
      params.offset = 0.5f;

      _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "conv8_2_mbox_priorbox"))
    {

      plugin::PriorBoxParameters params =
          { 0 };
      float minSize[1] =
          { 213.0f };
      float maxSize[1] =
          { 264.0f };
      float aspectRatios[1] =
          { 2.0f };
      params.minSize = (float*)minSize;
      params.maxSize = (float*)maxSize;
      params.aspectRatios = (float*)aspectRatios;
      params.numMinSize = 1;
      params.numMaxSize = 1;
      params.numAspectRatios = 1;
      params.flip = true;
      params.clip = false;
      params.variance[0] = 0.10000000149;
      params.variance[1] = 0.10000000149;
      params.variance[2] = 0.20000000298;
      params.variance[3] = 0.20000000298;
      params.imgH = 0;
      params.imgW = 0;
      params.stepH = 100.0f;
      params.stepW = 100.0f;
      params.offset = 0.5f;

      _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "conv9_2_mbox_priorbox"))
    {
      plugin::PriorBoxParameters params =
          { 0 };

      float minSize[1] =
          { 264.0f };
      float maxSize[1] =
          { 315.0f };
      float aspectRatios[1] =
          { 2.0f };
      params.minSize = (float*)minSize;
      params.maxSize = (float*)maxSize;
      params.aspectRatios = (float*)aspectRatios;
      params.numMinSize = 1;
      params.numMaxSize = 1;
      params.numAspectRatios = 1;
      params.flip = true;
      params.clip = false;
      params.variance[0] = 0.10000000149;
      params.variance[1] = 0.10000000149;
      params.variance[2] = 0.20000000298;
      params.variance[3] = 0.20000000298;
      params.imgH = 0;
      params.imgW = 0;
      params.stepH = 300.0f;
      params.stepW = 300.0f;
      params.offset = 0.5f;

      _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "detection_out")
        || !strcmp(layerName, "detection_out2")
            )
    {
      /*
       bool 	shareLocation
       bool 	varianceEncodedInTarget
       int 	backgroundLabelId
       int 	numClasses
       int 	topK
       int 	keepTopK
       float 	confidenceThreshold
       float 	nmsThreshold
       CodeType_t 	codeType
       */
      plugin::DetectionOutputParameters params =
          { 0 };
      params.numClasses = 21;
      params.shareLocation = true;
      params.varianceEncodedInTarget = false;
      params.backgroundLabelId = 0;
      params.keepTopK = 200;
      params.codeType = CENTER_SIZE;
      params.nmsThreshold = 0.45;
      params.topK = 400;
      params.confidenceThreshold = 0.5;

      _nvPlugins[layerName] = plugin::createSSDDetectionOutputPlugin(params);

      return _nvPlugins.at(layerName);

    }
    else if (

    !strcmp(layerName, "conv4_3_norm_mbox_conf_flat")
        || !strcmp(layerName, "conv4_3_norm_mbox_loc_flat")
        || !strcmp(layerName, "fc7_mbox_loc_flat")
        || !strcmp(layerName, "fc7_mbox_conf_flat")
        || !strcmp(layerName, "conv6_2_mbox_conf_flat")
        || !strcmp(layerName, "conv6_2_mbox_loc_flat")
        || !strcmp(layerName, "conv7_2_mbox_conf_flat")
        || !strcmp(layerName, "conv7_2_mbox_loc_flat")
        || !strcmp(layerName, "conv8_2_mbox_conf_flat")
        || !strcmp(layerName, "conv8_2_mbox_loc_flat")
        || !strcmp(layerName, "conv9_2_mbox_conf_flat")
        || !strcmp(layerName, "conv9_2_mbox_loc_flat")
        || !strcmp(layerName, "mbox_conf_flatten")

        )
    {

      _nvPlugins[layerName] = (plugin::INvPlugin*)(new FlattenLayer());

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "mbox_conf_reshape"))
    {

      _nvPlugins[layerName] = (plugin::INvPlugin*)new ReshapeLayer<21>();

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "mbox_loc"))
    {

      _nvPlugins[layerName] = plugin::createConcatPlugin(1, false);

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "mbox_conf"))
    {

      _nvPlugins[layerName] = plugin::createConcatPlugin(1, false);

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "mbox_priorbox"))
    {

      _nvPlugins[layerName] = plugin::createConcatPlugin(2, false);

      return _nvPlugins.at(layerName);

    }
    else
    {

      assert(0);

      return nullptr;

    }

  }

  // deserialization plugin implementation
  IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
  {
    if (!strcmp(layerName, "conv4_3_norm"))
    {

      _nvPlugins[layerName] = plugin::createSSDNormalizePlugin(serialData, serialLength);

      return _nvPlugins.at(layerName);

    }
    else if (
    !strcmp(layerName, "conv4_3_norm_mbox_loc_perm")
        || !strcmp(layerName, "conv4_3_norm_mbox_conf_perm")
        || !strcmp(layerName, "fc7_mbox_loc_perm")
        || !strcmp(layerName, "fc7_mbox_conf_perm")
        || !strcmp(layerName, "conv6_2_mbox_loc_perm")
        || !strcmp(layerName, "conv6_2_mbox_conf_perm")
        || !strcmp(layerName, "conv7_2_mbox_loc_perm")
        || !strcmp(layerName, "conv7_2_mbox_conf_perm")
        || !strcmp(layerName, "conv8_2_mbox_loc_perm")
        || !strcmp(layerName, "conv8_2_mbox_conf_perm")
        || !strcmp(layerName, "conv9_2_mbox_loc_perm")
        || !strcmp(layerName, "conv9_2_mbox_conf_perm")
            )
    {

      _nvPlugins[layerName] = plugin::createSSDPermutePlugin(serialData, serialLength);

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "conv4_3_norm_mbox_priorbox")
        || !strcmp(layerName, "fc7_mbox_priorbox")
        || !strcmp(layerName, "conv6_2_mbox_priorbox")
        || !strcmp(layerName, "conv7_2_mbox_priorbox")
        || !strcmp(layerName, "conv8_2_mbox_priorbox")
        || !strcmp(layerName, "conv9_2_mbox_priorbox")
            )
    {

      _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(serialData, serialLength);

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "detection_out")
        || !strcmp(layerName, "detection_out2")
            )
    {

      _nvPlugins[layerName] = plugin::createSSDDetectionOutputPlugin(serialData, serialLength);

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "mbox_conf_reshape"))
    {

      _nvPlugins[layerName] = (plugin::INvPlugin*)new ReshapeLayer<21>(serialData, serialLength);

      return _nvPlugins.at(layerName);

    }
    else if (

    !strcmp(layerName, "conv4_3_norm_mbox_conf_flat")
        || !strcmp(layerName, "conv4_3_norm_mbox_loc_flat")
        || !strcmp(layerName, "fc7_mbox_loc_flat")
        || !strcmp(layerName, "fc7_mbox_conf_flat")
        || !strcmp(layerName, "conv6_2_mbox_conf_flat")
        || !strcmp(layerName, "conv6_2_mbox_loc_flat")
        || !strcmp(layerName, "conv7_2_mbox_conf_flat")
        || !strcmp(layerName, "conv7_2_mbox_loc_flat")
        || !strcmp(layerName, "conv8_2_mbox_conf_flat")
        || !strcmp(layerName, "conv8_2_mbox_loc_flat")
        || !strcmp(layerName, "conv9_2_mbox_conf_flat")
        || !strcmp(layerName, "conv9_2_mbox_loc_flat")
        || !strcmp(layerName, "mbox_conf_flatten")
            )
    {

      _nvPlugins[layerName] = (plugin::INvPlugin*)(new FlattenLayer(serialData, serialLength));

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "mbox_loc"))
    {

      _nvPlugins[layerName] = plugin::createConcatPlugin(serialData, serialLength);

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "mbox_conf"))
    {

      _nvPlugins[layerName] = plugin::createConcatPlugin(serialData, serialLength);

      return _nvPlugins.at(layerName);

    }
    else if (!strcmp(layerName, "mbox_priorbox"))
    {

      _nvPlugins[layerName] = plugin::createConcatPlugin(serialData, serialLength);

      return _nvPlugins.at(layerName);

    }
    else
    {

      assert(0);

      return nullptr;

    }

  }

  void destroyPlugin()
  {

    for (auto it = _nvPlugins.begin(); it != _nvPlugins.end(); ++it)
    {

      it->second->destroy();

      _nvPlugins.erase(it);

    }

  }

private:

  std::map<std::string, plugin::INvPlugin*> _nvPlugins;

};

#endif
