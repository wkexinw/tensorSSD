#ifndef __LOGGER_H__
#define __LOGGER_H__
#include <cassert>
#include <iostream>
#include <fstream>
#include <cstring>
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;

using namespace nvcaffeparser1;

using namespace plugin;

class Logger: public ILogger
{

private:
  std::ofstream info_log;

public:

  Logger(const char * path)
  {
    info_log.open(path);
  }

  ~Logger()
  {
    info_log << std::endl;

    info_log.close();
  }

  void log(Severity severity, const char* msg) override
  {

    // suppress info-level messages
    //if (severity == Severity::kINFO) return;

    switch (severity)
    {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }

    std::cerr << msg << std::endl;

  }
};

#endif
