#ifndef IDEA_DNN_INFER_INFERCONFIG
#define IDEA_DNN_INFER_INFERCONFIG

#include <cstddef>
#include <string>

namespace idea::dnn::infer {
	enum class EngineType {
		ENGINE_UNKNOWNED,
		ENGINE_LIBTORCH,
		ENGINE_ONNX,
	};

	enum class DeviceType {
		DEVICE_UNKNOWNED,
		DEVICE_CPU,
		DEVICE_GPU
	};

	enum class ScalarType {
		SCALAR_FLOAT32,
		SCALAR_INT8,
		SCALAR_FLOAT16,
		SCALAR_BFLOAT16,
	};

	enum class DimensionType : int64_t {
		TYPE_BATCH = 0,
		TYPE_HEIGHT = 1,
		TYPE_WIDTH = 2,
		TYPE_CHANNEL = 3,
	};

	struct InferConfig {
		EngineType m_engine = EngineType::ENGINE_UNKNOWNED;
		DeviceType m_device = DeviceType::DEVICE_UNKNOWNED;
		ScalarType m_scalar = ScalarType::SCALAR_FLOAT32;
		std::string m_modelPath = "";
		std::size_t m_width = 20;
		std::size_t m_height = 1;
		std::size_t m_channel = 1;
		std::array<DimensionType, 4> m_dimension
			= { DimensionType::TYPE_BATCH, DimensionType::TYPE_CHANNEL, DimensionType::TYPE_HEIGHT, DimensionType::TYPE_WIDTH};
	};
}

#endif
