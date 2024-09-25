#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <filesystem>

int main() {
	torch::Device device(torch::kCUDA);
	std::string modelPath = "../examples/test_model.pt";

	if (!std::filesystem::exists(modelPath)) {
        std::cerr << "We can't find model file." << "\n";
        return -1;
    }

	torch::jit::script::Module module;
	try {
		std::cout << modelPath << "\n";
		module = torch::jit::load(modelPath, device);
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what();
		return -1;
	}

	std::cout << "No error!\n";

	std::string imgPath = "../examples/deer.png";
	auto img = cv::imread(imgPath);
	if (img.empty()) {
		std::cerr << "Image Error!\n";
		return -1;
	}

	cv::resize(img, img, cv::Size(32, 32));
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	img.convertTo(img, CV_32F, 1.0 / 255);
	auto tensor = torch::from_blob(img.data, { 1, img.rows, img.cols, 3 });
	tensor = tensor.permute({ 0, 3, 1, 2 }); // [H, W, C] -> [C, H, W]
	std::cout << "Tensor Dimention: " << tensor.sizes() << std::endl;
	tensor = tensor.contiguous().to(torch::kFloat).cuda();

	try {
		torch::Tensor output = module.forward({ tensor }).toTensor();
		std::cout << "Results: " << output << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what();
		return -1;
	}

	return 0;
}