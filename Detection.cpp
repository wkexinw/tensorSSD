/*
 * Detection.cpp
 *
 *  Created on: Mar 26, 2018
 *      Author: kevin
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>

#include "Detection.hpp"

Detection::Detection(detectionParams& detectorParam)
{
  m_isDecResolutionFound = detectorParam.isDecResolutionFound;
  m_inputQueue = detectorParam.rgbInputQueue;
//  m_yuvInputQueue = detectorParam.yuvInputQueue;
  m_isDectDoneForCurrentFrame = detectorParam.isDectDoneForCurrentFrame;
  m_bufNum = 0;
  m_detectedRectQueue = detectorParam.detectedRectQueue;
  m_detectionThread = detectorParam.detectionThread;
  m_imageWidth = detectorParam.imageWidth;
  m_imageHeight = detectorParam.imageHeight;
  m_cachefile = detectorParam.cachefile;
  m_confidence = detectorParam.confidence;

  setDetectorParams(detectorParam);

  printf("----->>>>> TRT : %d x %d\n", *m_imageWidth, *m_imageHeight);

//  jpegenc = NvJPEGEncoder::createJPEGEncoder("jpenenc");

  tensorSSD = new TensorSSD();

  tensorSSD->initialize(*m_imageWidth, *m_imageHeight);

  if (!tensorSSD->modelCacheExists(m_cachefile.c_str()))
  {

    printf("----->>>>> TRT : No cache model file found. Starting converting ...\n");

    bool convertDone = false;

    convertDone = tensorSSD->convertCaffeSSDModel(detectorParam.deployfile, detectorParam.modelfile, m_cachefile.c_str());

    if (!convertDone)
    {

      printf("\033[31,32m----->>>>> TRT : Caffe model convert error ...\033[0m\n");

      return;

    }

  }

  printf("----->>>>> TRT : Cache model file found. Starting loading ...\n");

  //load model cached file & get cached file size. 0 for failed.
  int cacheSize = tensorSSD->loadTensorSSDModel(m_cachefile.c_str());

  printf("----->>>>> TRT : Cache model file loaded %d bytes.\n", cacheSize);

}

void saveRGBA(unsigned char* data)
{

  FILE *fp = fopen("output.rgba", "wb");

  fwrite(data, 1280 * 720 * 4, 1, fp);

  fflush(fp);

  fclose(fp);

  exit(0);
}

bool Detection::setDetectorParams(detectionParams& detectorParam)
{

  return true;
}

int Detection::write_video_frame(ofstream * stream, NvBuffer* buffer)
{
  uint32_t i, j;
  char *data;

  for (i = 0; i < buffer->n_planes; i++)
  {
    NvBuffer::NvBufferPlane &plane = buffer->planes[i];
    size_t bytes_to_write = plane.fmt.bytesperpixel * plane.fmt.width;

    data = (char *)plane.data;
    for (j = 0; j < plane.fmt.height; j++)
    {
      stream->write(data, bytes_to_write);
      if (!stream->good())
        return -1;
      data += plane.fmt.stride;
    }
  }
  return 0;
}

std::mutex g_lock;

void Detection::detectionProcessLoop()
{

  NvBuffer *rgbaBuffer;

  NvBuffer *yuvBuffer;

  RGBBuffer rgba_buffer;

  YUVBuffer yuv_buffer;

  std::vector < cv::Rect > rectList;

  tensorSSD->prepareInference(m_confidence); //confidence 0~1.0 //default 0.9

  //wait until resolution has been found in decoder
  sem_wait(m_isDecResolutionFound);

  detectedRectBuffer bbox;

  while (1)
  {
//    yuv_buffer = m_yuvInputQueue->pop();
//
//    if (!yuv_buffer.isDetectionFrame)
//      continue;
//
//    yuvBuffer = yuv_buffer.buffer;
	rgba_buffer = m_inputQueue->pop();
	rgbaBuffer = rgba_buffer.buffer;

	tensorSSD->imageInference(rgbaBuffer, &rectList);


    bbox.g_rect = rectList;

    bbox.g_rect_num = rectList.size();

    m_detectedRectQueue->push(bbox);

    //saveRGBA(buffer->planes[0].data);

//#ifdef DEBUG
//
//    cv::Rect rect;
//
//    for (uint8_t i = 1; i < rectList.size(); i++)
//    {
//      rect = rectList[i];
//
//      printf("\033[32;43m[TensorSSD] Object detected %d: x = %d y = %d width = %d height = %d \033[0m\n", i, rect.x, rect.y, rect.width, rect.height);
//
//      out_buf_size = rect.width * rect.height * 3 / 2;
//
//      unsigned char *out_buf = new unsigned char[out_buf_size];
//
//      jpegenc->setCropRect(rect.x, rect.y, rect.width, rect.height);
//
//      jpegenc->encodeFromFd(yuvBuffer->planes[0].fd, JCS_YCbCr, &out_buf, out_buf_size);
//
//      char fname[80];
//
//      sprintf(fname, "test_%d.jpg", i - 1);
//
//      out_file = new ofstream(fname);
//
//      out_file->write((const char*)out_buf, out_buf_size);
//
//      out_file->close();
//
//      //memset(out_buf,0,out_buf_size);
//
//      delete[] out_buf;
//
//    }
//
//#endif

    sem_post(m_isDectDoneForCurrentFrame);

  }

}

Detection::~Detection()
{

}
