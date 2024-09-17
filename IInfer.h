#ifndef IDEA_DNN_INFER_IINFER
#define IDEA_DNN_INFER_IINFER

#include <string>
#include <opencv2/opencv.hpp>
#include "InferConfig.h"

namespace idea::dnn::infer {
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
