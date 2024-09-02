#ifndef IDEA_INFERDNN_INFERLIBTORCH
#define IDEA_INFERDNN_INFERLIBTORCH

#include "AInfer.h"
#include <torch/script.h>
#include <torch/torch.h>

namespace idea::inferdnn {
	class InferLibtorch : public AInfer
	{
	public:
		InferLibtorch() = default;
		virtual ~InferLibtorch() = default;

	public:
		bool onCreate(const std::string& modelPath, const std::string& config) override;
		bool onProcess(const cv::Mat& mat) override;
		bool onDestroy() override;

	private:
		torch::jit::script::Module							m_module;
	};
}

#endif
