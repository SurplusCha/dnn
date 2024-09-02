#ifndef IDEA_INFERDNN_AINFER
#define IDEA_INFERDNN_AINFER

#include <string>
#include <opencv2/opencv.hpp>

namespace idea::inferdnn {
	class AInfer 
	{
	public:
		AInfer() = default;
		virtual ~AInfer() = default;

	public:
		bool create(const std::string& modelPath, const std::string& config);
		bool process(const cv::Mat& mat);
		bool destroy();

	public:
		virtual bool onCreate(const std::string& modelPath, const std::string& config) = 0;
		virtual bool onProcess(const cv::Mat& mat) = 0;
		virtual bool onDestroy() = 0;

	private:
		bool setCreate = false;
	};
}

#endif
