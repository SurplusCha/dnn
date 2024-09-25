//
// Created by systembug on 24. 9. 24.
//

#include <boost/log/trivial.hpp>
#include <boost/algorithm/string.hpp>
#include <memory>
#include <filesystem>
#include "ConfigManager.h"
#include "yaml-cpp/yaml.h"
#include "InferConfig.h"

namespace idea::dnn::infer {
    InferConfig* ConfigManager::parse(const std::string& path)
    {
        if (!std::filesystem::exists(path)) {
            BOOST_LOG_TRIVIAL(error) << "This path is invalid.";
            return nullptr;
        }

        YAML::Node configFile;
        try {
            configFile = YAML::LoadFile(path);
        }
        catch (const std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << e.what();
            return nullptr;
        }

        auto config = std::make_shared<InferConfig>();
        try {
            auto engineType = configFile["engine"].as<std::string>();
            boost::to_lower(engineType);
            if (engineType == "torch") {
                config->m_engine = EngineType::ENGINE_LIBTORCH;
            } else if (engineType == "onnx") {
                config->m_engine = EngineType::ENGINE_ONNX;
            } else {
                config->m_engine = EngineType::ENGINE_UNKNOWNED;
            }

            auto deviceType = configFile["device"].as<std::string>();
            boost::to_lower(deviceType);
            if (deviceType == "gpu") {
                config->m_device = DeviceType::DEVICE_GPU;
            } else if (deviceType == "cpu") {
                config->m_device = DeviceType::DEVICE_CPU;
            } else {
                config->m_device = DeviceType::DEVICE_UNKNOWNED;
            }

            auto scalarType = configFile["scalar"].as<std::string>();
            boost::to_lower(scalarType);
            if (scalarType == "float" || scalarType == "float32") {
                config->m_scalar = ScalarType::SCALAR_FLOAT;
            }
            else if (scalarType == "float16") {
                config->m_scalar = ScalarType::SCALAR_FLOAT16;
            }
            else if (scalarType == "int8") {
                config->m_scalar = ScalarType::SCALAR_INT8;
            }
            else if (scalarType == "bfloat16") {
                config->m_scalar = ScalarType::SCALAR_BFLOAT16;
            }
            else {
                config->m_scalar = ScalarType::SCALAR_FLOAT;
            }

            config->m_modelPath = configFile["model_path"].as<std::string>();
            config->m_width = configFile["width"].as<std::size_t>();
            config->m_height = configFile["height"].as<std::size_t>();
            config->m_channel = configFile["channel"].as<std::size_t>();

            auto dimensionArr = configFile["dimension"];
            for (auto i = 0; i < dimensionArr.size(); ++i)
                config->m_dimension[i] = static_cast<DimensionType>(dimensionArr[i].as<int64_t>());
        }
        catch (const std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << e.what();
            return nullptr;
        }

        return config.get();
    }
}