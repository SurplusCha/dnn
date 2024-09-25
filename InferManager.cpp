#include <boost/log/trivial.hpp>
#include <filesystem>
#include "InferManager.h"
#include "InferLibtorch.h"
#include "InferONNX.h"

namespace idea::dnn::infer {
	bool InferManager::create(const std::string& configPath)
	{
		// TODO: get information from config file
		auto config = m_configManager.parse(configPath);
        if (config == nullptr) {
            BOOST_LOG_TRIVIAL(error) << "Configuration to infer a model cannot be created.";
            return false;
        }

		switch (config->m_engine) {
			case EngineType::ENGINE_LIBTORCH:
				m_infer = std::make_unique<InferLibtorch>();
				break;
			case EngineType::ENGINE_ONNX:
				m_infer = std::make_unique<InferONNX>();
				break;
			case EngineType::ENGINE_UNKNOWNED:
			default:
				m_infer = std::make_unique<InferNothing>();
				break;
		}

		if (!m_infer->onCreate(*config))
			return false;

		return true;
	}

	bool InferManager::process(const cv::Mat& mat)
	{
		return m_infer->onProcess(mat);
	}

	bool InferManager::destroy()
	{
		return m_infer->onDestroy();
	}
}