#include "InferONNX.h"

namespace idea::dnn::infer {
	bool InferONNX::onCreate(const InferConfig& config)
	{
		return true;
	}

	bool InferONNX::onProcess(const cv::Mat& mat)
	{
		return true;
	}

	bool InferONNX::onDestroy()
	{
		return true;
	}
}