SRCS:= $(wildcard *.cpp)
SRCS+= $(wildcard *.cu)

OBJS := $(SRCS:.cpp=.o)
OBJS2+= $(SRCS:.cu=.o)

CUCC = /usr/local/cuda-9.0/bin/nvcc

CUFLAGS = -m64 -ccbin g++

CPPFLAGS += -std=c++11 \
	-I"$(CUDA_PATH)/targets/aarch64-linux/include" -I"./" \
	-I"../common" \
	-I"/usr/include/glib-2.0" \
	-I"/usr/include/libxml2" \
	-I"/usr/include"


LDFLAGS += \
	-L"$(CUDA_PATH)/targets/aarch64-linux/lib" \
	-lnvinfer -lnvcaffe_parser -lnvinfer_plugin -lcudnn -lcublas -lnvToolsExt \
	-lcudart -lglog -lnvgraph -lglib-2.0 -lgobject-2.0

all: $(OBJS) $(OBJS2)

%.o: %.cpp
	@echo "Compiling: $<"
	$(CPP) $(CPPFLAGS) -c $<

%.o: %.cu
	$(CUCC) $(CUFLAGS) -c -o $@ $<
clean:
	$(AT)rm -rf *.o
