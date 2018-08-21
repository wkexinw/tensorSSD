/*
 * Detection.hpp
 *
 *  Created on: Mar 26, 2018
 *      Author: kevin
 */

#ifndef __DETECTION_H__
#define __DETECTION_H__
#include <pthread.h>
#include <semaphore.h>
#include <string>
#include<time.h>
#include <iostream>
#include <fstream>
//#include <NvJpegEncoder.h>
#include "Queue.h"
#include "bufferStruct.hpp"
#include "TensorSSD.hpp"

using namespace std;

struct detectionParams
{
  int detectorModelIndex;
  string deployfile;
  string modelfile;
  string cachefile;
  int interval;
  Queue<RGBBuffer>* rgbInputQueue;
  Queue<YUVBuffer>* yuvInputQueue;
  Queue<detectedRectBuffer>* detectedRectQueue;
  sem_t* isDectDoneForCurrentFrame;
  sem_t* isDecResolutionFound;
  pthread_t* detectionThread;
  int* imageWidth;
  int* imageHeight;
  float confidence;
};

struct detectionOutputParams
{
  int netWidth;
  int netHeight;
};

class Detection
{
public:
  Detection(detectionParams&);
  virtual ~Detection();
  void runDetectionThread()
  {
    pthread_create(m_detectionThread, NULL, _detectionProcessLoop, (void *)this);
  }

  void getDectecterParams(detectionOutputParams& params)
  {
    params.netWidth = m_netWidth;
    params.netHeight = m_netHeight;
  }

private:

//  NvJPEGEncoder *jpegenc;
//
//  unsigned long out_buf_size;
//
//  std::ofstream * out_file;

  TensorSSD* tensorSSD;
  string m_cachefile;
  float m_confidence;
  int write_video_frame(ofstream * stream, NvBuffer* buffer);
  bool setDetectorParams(detectionParams&);
  void detectionProcessLoop();
  static void* _detectionProcessLoop(void* arg)
  {
    ((Detection *)arg)->detectionProcessLoop();
  }

  Queue<RGBBuffer>* m_inputQueue;
//  Queue<YUVBuffer>* m_yuvInputQueue;
  Queue<detectedRectBuffer>* m_detectedRectQueue;
  sem_t* m_isDectDoneForCurrentFrame;
  sem_t* m_isDecResolutionFound;
  int m_bufNum;
  int m_netWidth;
  int m_netHeight;
  pthread_t* m_detectionThread;
  int* m_imageWidth;
  int* m_imageHeight;

};

#endif
