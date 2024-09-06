#ifndef IDEA_INFERDNN_INFERNOTHING
#define IDEA_INFERDNN_INFERNOTHING

#include "IInfer.h"

namespace idea::inferdnn {
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
