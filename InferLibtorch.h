#ifndef IDEA_INFERDNN_INFERLIBTORCH
#define IDEA_INFERDNN_INFERLIBTORCH

#include "IInfer.h"
#include <torch/script.h>
#include <torch/torch.h>
#include "InferConfig.h"

namespace idea::inferdnn {
	class InferLibtorch : public IInfer
	{
	public:
		InferLibtorch() = default;
		virtual ~InferLibtorch() = default;

	public:
		bool onCreate(const InferConfig& config) override;
		bool onProcess(const cv::Mat& mat) override;
		bool onDestroy() override;

	private:
		torch::jit::script::Module							m_module;
		c10::DeviceType 									m_deviceType;
		c10::ScalarType										m_scalarType;
		c10::IntArrayRef									m_dimensionType;
	};
}

#endif
