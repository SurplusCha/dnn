#ifndef IDEA_DNN_INFER_INFERMANAGER
#define IDEA_DNN_INFER_INFERMANAGER

#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include "InferNothing.h"

namespace idea::dnn::infer {
	class InferManager
	{
	public:
		InferManager() = default;
		~InferManager() = default;

	public:
		bool create(const std::string& configPath);
		bool process(const cv::Mat& mat);
		bool destroy();

	private:
		std::unique_ptr<IInfer>	m_infer = std::make_unique<InferNothing>();
	};
}

#endif

