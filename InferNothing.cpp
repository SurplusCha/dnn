#include <boost/log/trivial.hpp>
#include "InferNothing.h"

namespace idea::dnn::infer {
	bool InferNothing::onCreate(const InferConfig& config)
	{
		BOOST_LOG_TRIVIAL(warning) << "InferNothing is created. This means that you didn't make anything which can infer model.";
		return true;
	}

	bool InferNothing::onProcess(const cv::Mat& mat)
	{
		BOOST_LOG_TRIVIAL(warning) << "Nothing can be inferred. Because this object is InferNothing.";
		return true;
	}

	bool InferNothing::onDestroy()
	{
		return true;
	}
}