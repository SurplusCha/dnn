#include "InferONNX.h"

namespace idea::inferdnn {
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