#include<torch/script.h>
#include<memory>
#include<cstddef>
#include<iostream>
#include<string>
#include<vector>
#include<torch/torch.h>
#include<assert.h>
#include<opencv2/opencv.hpp>
#include<unordered_map>
#include<string>
#include<sstream>
#include<istream>
struct Net: torch::nn::Module
{
	Net():conv1(torch::nn::Conv2dOptions(1,10,5)),conv2(torch::nn::Conv2dOptions(10,20,5)),fc1(320,50),fc2(50,10)
	{
		register_module("conv1",conv1);
		register_module("conv2",conv2);
		register_module("fc1",fc1);
		register_module("fc2",fc2);	
	}
	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(torch::max_pool2d(conv1->forward(x),2));
		x = torch::relu(torch::max_pool2d(conv2->forward(x),2));

		x = x.view({-1,320});
		x = torch::relu(fc1->forward(x));
		//x = torch::dropout(x,0.5,is_training());
		x = fc2->forward(x);
		return torch::log_softmax(x,1);
	}
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
};

//this part is using the options about this system
struct Options{
	std::string data_root{"data"};
	int32_t batch_size{64};
	int32_t epochs{10};
	double lr{0.045};
	double momentum{0.9};
	bool no_cuda{false};
	int32_t seed{1};
	int32_t test_batch_size{1000};
	int32_t log_interval{10};
};
template <typename DataLoader>
void train(
    int32_t epoch,
    const Options& options,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::SGD& optimizer,
    size_t dataset_size) {
  model.train();
	model.to(device);
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
		std::cout<<output[0]<<std::endl;
    auto loss = torch::nll_loss(output, targets);
    loss.backward();
    optimizer.step();

    if (batch_idx++ % options.log_interval == 0) {
      std::cout << "Train Epoch: " << epoch << " ["
                << batch_idx * batch.data.size(0) << "/" << dataset_size
                << "]\tLoss: " << loss.template item<float>() << std::endl;
    }
  }
}
template <typename DataLoader>
void test(
    std::shared_ptr<torch::jit::script::Module> &model,
    torch::Device device,
    DataLoader &data_loader,
    size_t dataset_size) {
  torch::NoGradGuard();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(data);
		at::Tensor output = model->forward(inputs).toTensor();
		output = torch::log_softmax(output,1);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     {},
                     Reduction::Mean)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::cout << "Test set: Average loss: " << test_loss
            << ", Accuracy: " << correct << "/" << dataset_size << std::endl;
}
void predict(std::shared_ptr<torch::jit::script::Module> &model,
			torch::Device device,
			std::vector<cv::Mat>& inputs,
			std::vector<long> &targets)
{

	model->to(device);
	size_t data_size = inputs.size();
	uint32_t predict = 0;
	for(int i =0;i<inputs.size();i++)
	{
			std::cout<<"Process: "<<i+1<<"/"<<data_size<<std::endl;
			torch::Tensor im = torch::from_blob(inputs[i].data,{1,inputs[i].rows,inputs[i].cols,3},torch::kByte);
			im = im.to(torch::kDouble);
			im = im.div(255.0).sub(torch::tensor({0.485,0.456,0.406})).div(torch::tensor({0.229,0.224,0.225}));

			im = im.permute({0,3,1,2});	
			
			im = im.to(torch::kFloat).to(device);

			auto output = model->forward({im}).toTensor();
			auto ind = output.max(1,true);
			auto max_index = std::get<1>(ind).item<long>();
			if(max_index == targets[i])
			{
				predict++;
			}

	}
	auto preds= 1.*predict/data_size;
	std::cout<<"Predict: "<<preds<<std::endl;


}
auto main(int argc, const char * argv[])->int{
	assert(argc == 4);
	torch::manual_seed(0);
	Options options;
	std::unordered_map<int,std::string> objs;
	objs[0] = "ants";
	objs[1] = "bees";
	std::string ants{argv[1]};
	std::string bees{argv[2]};
	
	//build the inputs,and target
	std::vector<cv::Mat> inputs;
	std::vector<long> targets;
	std::ifstream fi(ants);
	std::string temp;
	while(std::getline(fi,temp))
	{
		cv::Mat temImage;
		cv::resize(cv::imread(temp),temImage,cv::Size(224,224));
		cv::cvtColor(temImage,temImage,cv::COLOR_BGR2RGB);
		inputs.push_back(std::move(temImage));
		targets.push_back(0);
	}
	fi.close();
	fi.open(bees);
	while(std::getline(fi,temp))
	{
		cv::Mat temImage;
		cv::resize(cv::imread(temp),temImage,cv::Size(224,224));
		cv::cvtColor(temImage,temImage,cv::COLOR_BGR2RGB);
		inputs.push_back(std::move(temImage));
		targets.push_back(1);
	}
	fi.close();
	torch::DeviceType device_type;
	if (torch::cuda::is_available() && !options.no_cuda) {
		std::cout << "CUDA available! Training on GPU" << std::endl;
		device_type = torch::kCUDA;
	} else {
		std::cout << "Training on CPU" << std::endl;
		device_type = torch::kCPU;
	}
	torch::Device device(device_type);
	std::shared_ptr<torch::jit::script::Module> model = torch::jit::load(argv[3]);
	predict(model,device,inputs,targets);
}
