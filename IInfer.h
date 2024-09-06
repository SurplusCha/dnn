#ifndef IDEA_INFERDNN_IINFER
#define IDEA_INFERDNN_IINFER

#include <string>
#include <opencv2/opencv.hpp>
#include "InferConfig.h"

namespace idea::inferdnn {
	class IInfer 
	{
	public:
		IInfer() = default;
		virtual ~IInfer() = default;

	public:
		virtual bool onCreate(const InferConfig& configPath) = 0;
		virtual bool onProcess(const cv::Mat& mat) = 0;
		virtual bool onDestroy() = 0;
	};
}

#endif
