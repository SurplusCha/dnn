#ifndef IDEA_INFERDNN_INFERONNX
#define IDEA_INFERDNN_INFERONNX

#include "IInfer.h"

namespace idea::inferdnn {
	class InferONNX : public IInfer
	{
	public:
		InferONNX() = default;
		virtual ~InferONNX() = default;

	public:
		bool onCreate(const InferConfig& config) override;
		bool onProcess(const cv::Mat& mat) override;
		bool onDestroy() override;
	};
}

#endif

