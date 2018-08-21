#ifndef __COMMON_H__
#define __COMMON_H__

#include <cassert>
#include <iostream>
#include <cudnn.h>
#include <cstring>
#include <cuda_runtime.h> 

#define CHECK(status)												\
{																	\
	if (status != 0)                                                \
	{                                                               \
		std::cout << "Cuda failure: " << cudaGetErrorString(status) \
				  << " at line " << __LINE__                        \
				  << std::endl;                                     \
		abort();                                                    \
	}                                                               \
}

#endif
