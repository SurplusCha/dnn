#ifndef IDEA_DNN_INFER_INFERLIBTORCH
#define IDEA_DNN_INFER_INFERLIBTORCH

#include "IInfer.h"
#include <torch/script.h>
#include <torch/torch.h>
#include "InferConfig.h"

namespace idea::dnn::infer {
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
		c10::DeviceType 									m_deviceType = torch::kCPU;
		c10::ScalarType										m_scalarType = torch::kFloat;
		std::array<std::size_t, 4>						    m_dimensionType = {0, 1, 2, 3};
	};
}

#endif
