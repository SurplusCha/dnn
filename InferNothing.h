#ifndef IDEA_DNN_INFER_INFERNOTHING
#define IDEA_DNN_INFER_INFERNOTHING

#include "IInfer.h"

namespace idea::dnn::infer {
	class InferNothing : public IInfer
	{
	public:
		InferNothing() = default;
		virtual ~InferNothing() = default;

	public:
		bool onCreate(const InferConfig& config) override;
		bool onProcess(const cv::Mat& mat) override;
		bool onDestroy() override;
	};
}

#endif
