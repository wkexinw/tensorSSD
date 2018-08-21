/*
 * Detection.cpp
 *
 *  Created on: Mar 26, 2018
 *      Author: kevin
 */

#include "Detection.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#define MODEL_CACHE_FILE "trtModel.cache"

Detection::Detection(detectionParams& detectorParam)
{

  m_isDecResolutionFound = detectorParam.isDecResolutionFound;
  m_inputQueue = detectorParam.rgbInputQueue;
  m_isDectDoneForCurrentFrame = detectorParam.isDectDoneForCurrentFrame;
  m_bufNum = 0;
  m_detectedRectQueue = detectorParam.detectedRectQueue;
  m_detectionThread = detectorParam.detectionThread;
  m_imageWidth = detectorParam.imageWidth;
  m_imageHeight = detectorParam.imageHeight;
  m_cachefile = detectorParam.cachefile;
  m_confidence = detectorParam.confidence;

  setDetectorParams(detectorParam);

  getDectecterParams();

  tensorSSD = new TensorSSD();

  tensorSSD->initialize(1280, 720);

  if (!tensorSSD->modelCacheExists(MODEL_CACHE_FILE))
  {

    printf("----->>>>> TRT : No cache model file found. Starting converting ...\n");

    tensorSSD->convertCaffeSSDModel(detectorParam.deployfile, detectorParam.modelfile, MODEL_CACHE_FILE);

  }
  else
  {

    printf("----->>>>> TRT : Cache model file found. Starting loading ...\n");

  }
  //load model cached file & get cached file size. 0 for failed.
  int cacheSize = tensorSSD->loadTensorSSDModel(MODEL_CACHE_FILE);

  printf("----->>>>> TRT : Cache model file loaded %d bytes.\n", cacheSize);

}

Detection::~Detection()
{

}

bool Detection::setDetectorParams(detectionParams& detectorParam)
{

  return true;
}

void Detection::getDectecterParams()
{
}

void Detection::detectionProcessLoop()
{

  struct v4l2_buffer *v4l2_buf;

  NvBuffer *buffer;

  vector < cv::Rect > rectList;

  tensorSSD->prepareInference(m_confidence); //confidence 0~1.0 //default 0.9

  //wait until resolution has been found in decoder
  sem_wait(m_isDecResolutionFound);

  while (1)
  {

    RGBBuffer trt_buffer;

    trt_buffer = m_inputQueue->pop();

    v4l2_buf = &trt_buffer.v4l2_buf;

    buffer = trt_buffer.buffer;

    //TO DO TensorSSD
    vector < cv::Rect > rectList;

    tensorSSD->imageInference((float4*)buffer->planes[0].data, &rectList);

    detectedRectBuffer bbox;

    bbox.g_rect = rectList;

    bbox.g_rect_num = rectList.size();

    m_detectedRectQueue->push(bbox);

    /*
     cv::Rect rect;

     for (uint8_t i = 0; i < rectList.size(); i++)
     {

     rect = rectList[i];

     printf("Object detected %d: x = %d y = %d width = %d height = %d\n",
     i, rect.x, rect.y, rect.width, rect.height);

     cv::rectangle(imgbgr, rect, cv::Scalar(255, 0, 0), 2, 1);

     }
     */

    sem_post(m_isDectDoneForCurrentFrame);

  }
}
