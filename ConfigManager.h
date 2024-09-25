//
// Created by systembug on 24. 9. 24.
//

#ifndef IDEA_DNN_INFER_CONFIGMANAGER
#define IDEA_DNN_INFER_CONFIGMANAGER

#include <string>
#include <memory>

namespace idea::dnn::infer {
    class InferConfig;
    class ConfigManager {
    public:
        ConfigManager() = default;
        ~ConfigManager() = default;

    public:
        std::shared_ptr<InferConfig> parse(const std::string& path);
    };
}


#endif //DNN_CONFIGMANAGER_H
