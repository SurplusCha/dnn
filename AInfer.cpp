#include "AInfer.h"
#include <filesystem>

namespace idea::inferdnn {
	bool AInfer::create(const std::string& modelPath, const std::string& config)
	{
		if (!std::filesystem::exists(modelPath))
			return false;

		if (!onCreate(modelPath, config))
			return false;

		setCreate = true;
	}

	bool AInfer::onProcess(const cv::Mat& mat)
	{
		if (setCreate)
			return onProcess(mat);
		else
			return false;
	}

	bool AInfer::destroy()
	{
		if (!onDestroy())
			return false;

		setCreate = false;
		return true;
	}
}