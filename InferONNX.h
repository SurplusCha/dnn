#ifndef IDEA_DNN_INFER_INFERONNX
#define IDEA_DNN_INFER_INFERONNX

#include "IInfer.h"

namespace idea::dnn::infer {
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

