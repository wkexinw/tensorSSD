SRCS:= $(wildcard *.cpp)
SRCS2:= $(wildcard ../*.cu)
OBJS := $(SRCS:.cpp=.o) 
OBJS2:= $(SRCS2:.cu=.o)
AT=@
CCX = g++

CUCC = /usr/local/cuda-9.0/bin/nvcc

CUFLAGS = -m64 -ccbin g++

CFLAGS = -Wall -std=c++11 -O3 -I /usr/local/cuda-9.0/include -I /usr/local/include \
	-I /usr/local/cuda-9.0/include  -I /usr/include/gstreamer-1.0 \
	-I /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include -I /usr/include/glib-2.0 \
	-I /usr/include/libxml2 -I /usr/lib/aarch64-linux-gnu/glib-2.0/include \
	-I /usr/include/qt4/QtCore/ -I /usr/include/qt4/ -I /usr/include/qt4/QtGui -I ../../common\
	-I /usr/include/qt4/Qt -I ../common -I ../ -D_REENTRANT -g 


LD_FLAGS = -L/usr/local/cuda-9.0/targets/x86_64-linux/lib -L/usr/local/lib \
	-L/usr/local/cuda-9.0/lib -L/usr/local/cuda-9.0/lib64 \
	-L/usr/lib/aarch64-linux-gnu -L/usr/lib/aarch64-linux-gnu \
	-L/usr/lib/aarch64-linux-gnu/gstreamer-1.0/ \
	-Wl,--start-group -lnvinfer -lnvcaffe_parser -lnvinfer_plugin -lcudnn -lcublas -lnvToolsExt \
	-lcudart -lglog -lnvgraph -lGL -lGLEW -lX11 -lQtCore -lglib-2.0 -lgobject-2.0 -lgstapp-1.0\
	-lgstreamer-1.0 -lQtGui -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc \
	-lopencv_videoio -lrt -ldl -lpthread

all:$(OBJS) $(OBJS2)
	$(AT) $(CCX) -o sample_SSD $^ $(LD_FLAGS)

%.o: %.cpp
	$(CCX) -c -o $@ $< $(CFLAGS)

%.o: %.cu
	$(CUCC) $(CUFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f ../*.o
