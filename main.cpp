#include <opencv2/opencv.hpp>
#include "InferManager.h"

int main() {
    idea::dnn::infer::InferManager inferManager;
    if (!inferManager.create("../examples/config.yaml")) {
        std::cerr << "InferManager Error!\n";
        return -1;
    }

    std::string imgPath = "../examples/deer.png";
    auto img = cv::imread(imgPath);
    if (img.empty()) {
        std::cerr << "Image Error!\n";
        return -1;
    }

    cv::resize(img, img, cv::Size(32, 32));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0 / 255);

    inferManager.process(img);
    inferManager.destroy();
    return 0;
}