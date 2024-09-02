#include "InferLibtorch.h"

namespace idea::inferdnn {
	bool InferLibtorch::onCreate(const std::string& modelPath, const std::string& config)
	{
		torch::Device device(torch::kCUDA);
		try {
			std::cout << modelPath << "\n";
			m_module = torch::jit::load(modelPath, device);
		}
		catch (const std::exception& e) {
			std::cerr << "Error: " << e.what();
			return false;
		}

		return true;
	}

	bool InferLibtorch::onProcess(const cv::Mat& mat)
	{
		return true;
	}

	bool InferLibtorch::onDestroy()
	{
		return true;
	}
}