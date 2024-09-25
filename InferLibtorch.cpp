#include <boost/log/trivial.hpp>
#include "InferLibtorch.h"
#include <filesystem>

namespace idea::dnn::infer {
	bool InferLibtorch::onCreate(const InferConfig& config)
	{
		switch (config.m_device) {
			case DeviceType::DEVICE_GPU:
				m_deviceType = torch::kCUDA;
				break;
			case DeviceType::DEVICE_CPU:
			case DeviceType::DEVICE_UNKNOWNED:
			default:
				m_deviceType = torch::kCPU;
				break;
		}

		switch (config.m_scalar) {
			case ScalarType::SCALAR_INT8:
				m_scalarType = torch::kInt8;
				break;
			case ScalarType::SCALAR_FLOAT16:
				m_scalarType = torch::kFloat16;
				break;
			case ScalarType::SCALAR_BFLOAT16:
				m_scalarType = torch::kBFloat16;
				break;
			case ScalarType::SCALAR_FLOAT32:
			default:
				m_scalarType = torch::kFloat;
				break;
		}

		torch::Device device(m_deviceType);
		try {
			BOOST_LOG_TRIVIAL(trace) << "Model Path: " << config.m_modelPath;
			m_module = torch::jit::load(config.m_modelPath, device);
		}
		catch (const std::exception& e) {
			BOOST_LOG_TRIVIAL(error) << "Error: " << e.what();
			return false;
		}

        std::transform(std::cbegin(config.m_dimension), std::cend(config.m_dimension),
                       std::begin(m_dimensionType), [](const DimensionType& value) {
            return static_cast<std::size_t>(value);
        });
        return true;
	}

	bool InferLibtorch::onProcess(const cv::Mat& mat)
	{
		auto tensor = torch::from_blob(mat.data, { 1, mat.rows, mat.cols, mat.channels()});
        auto dimensionType = c10::IntArrayRef(reinterpret_cast<const int64_t*>(m_dimensionType.data()), 4);
		tensor = tensor.permute(dimensionType);
		tensor = tensor.contiguous().to(m_deviceType, m_scalarType, false, false);		

		try {
			torch::Tensor output = m_module.forward({ tensor }).toTensor();
            BOOST_LOG_TRIVIAL(info) << output;
		}
		catch (const std::exception& e) {
			BOOST_LOG_TRIVIAL(error) << e.what();
			return -1;
		}

		return true;
	}

	bool InferLibtorch::onDestroy()
	{
		return true;
	}
}