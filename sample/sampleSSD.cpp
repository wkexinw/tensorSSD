#include <algorithm>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <memory>
#include <cstring>

//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
//TensorSSD
#include "TensorSSD.hpp"
//camera
#include "gstCamera.h"

using namespace std;
using namespace cv;

#define DEFAULT_CAMERA -1

static int CAMERA_W = 320;

static int CAMERA_H = 240;

static int CAMERA_PIXEL_DEPTH = 1;

#define SSD_MODEL "/home/nvidia/wangkx/Minibox/data/Model/SSD/SSD.caffemodel"
#define SSD_PROTOTXT "/home/nvidia/wangkx/Minibox/data/Model/SSD/SSD.prototxt"
#define MODEL_CACHE_FILE "./trtModel.cache"

bool signal_recieved = false;

void sig_handler(int signo)
{

  if (signo == SIGINT)
  {

    signal_recieved = true;

  }

}

int main(int argc, char** argv)
{

  if (signal(SIGINT, sig_handler) == SIG_ERR)
  {

    printf("\ncan't catch SIGINT\n");

  }

  gstCamera* camera = gstCamera::Create(CAMERA_W,CAMERA_H,DEFAULT_CAMERA);

  if (!camera)
  {

    printf("failed to initialize video device\n");

    return 0;

  }

  CAMERA_H = camera->GetHeight();
  CAMERA_W = camera->GetWidth();
  CAMERA_PIXEL_DEPTH = camera->GetPixelDepth();

  printf("successfully initialized video device\n");
  printf("camera width:  %u\n", CAMERA_W);
  printf("camera height:  %u\n", CAMERA_H);
  printf("camera depth:  %u (bpp)\n\n", CAMERA_PIXEL_DEPTH);

  if (!camera->Open())
  {

    printf("failed to open camera for streaming\n");

    return 0;

  }

  //create TensorSSD
  TensorSSD* tensorSSD = new TensorSSD();
  //initialize with source dimistion
  tensorSSD->initialize(CAMERA_W, CAMERA_H);

  if (!tensorSSD->modelCacheExists(MODEL_CACHE_FILE))
  {

    tensorSSD->convertCaffeSSDModel(SSD_PROTOTXT, SSD_MODEL, MODEL_CACHE_FILE);

  }
  //load model cached file & get cached file size. 0 for failed.
  int cacheSize = tensorSSD->loadTensorSSDModel(MODEL_CACHE_FILE);

  vector < Rect > rectList;

  tensorSSD->prepareInference(0.9); //confidence 0~1.0

  struct  timeval start;
  struct  timeval end;
  unsigned  long diff;


  while (!signal_recieved)
  {

    void* imgCPU = NULL;
    void* imgCUDA = NULL;
    void* imgRGBA = NULL;

    if (!camera->Capture(&imgCPU, &imgCUDA, 1000))
      printf("failed to capture frame\n");

    cv::Mat imcuda(CAMERA_H * 3 / 2, CAMERA_W, CV_8UC1, (unsigned char*)imgCUDA);

    cv::Mat imgbgr;

    cv::cvtColor(imcuda, imgbgr, CV_YUV2BGR_NV12);

    if (!camera->ConvertRGBA(imgCUDA, &imgRGBA))
      printf("failed to convert from NV12 to RGBA\n");
  gettimeofday(&start,NULL);
    //do inference with an image(RGBA)
    tensorSSD->imageInference((float4*)imgRGBA, &rectList);
  gettimeofday(&end,NULL);

  diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;

  printf("\033[32;34m Cost Time (imageInference) %lu \033[0m\n",diff);
    Rect rect;

    for (uint8_t i = 0; i < rectList.size(); i++)
    {

      rect = rectList[i];

      printf("Object detected %d: x = %d y = %d width = %d height = %d\n",
          i, rect.x, rect.y, rect.width, rect.height);

      cv::rectangle(imgbgr, rect, cv::Scalar(255, 0, 0), 2, 1);

    }

    rectList.clear();

    cv::imshow("tracker", imgbgr);

    cv::waitKey(1);

  }

  tensorSSD->shutdown();

  if (camera != NULL)
  {

    delete camera;

    camera = NULL;

  }

  cout << "Done.\r\n" << endl;

  return 0;

}

