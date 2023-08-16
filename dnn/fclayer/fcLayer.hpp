#ifndef _full_connected_layer__
#define _full_connected_layer__

/**********************************
* Header for Neural Network layer *
**********************************/
#include "../neural/neural.hpp"

class fcLayer
{
public:
	fcLayer() : size(0) {};
	fcLayer(int num, Actv actv);
	//float* generate_output();
	size_t getSize() { return size; }
	std::vector<float> backCompute(float loss);
	std::vector<std::vector<float>> backCompute(std::vector<std::vector<float>> loss);

	// activation function
	void sigmoid();
	void relu();
	void tanh();
	std::vector<neural> neurals;

private:
	size_t size = 0;
	Actv mode;
};

fcLayer::fcLayer(const int num, Actv actv) {
	mode = actv;
	std::vector<neural> temp(num);
	neurals = temp;
	size = neurals.size();
}

// 单神经元输出层使用的反向传播函数
std::vector<float> fcLayer::backCompute(float loss)
{
	std::vector<float> results;
	for (size_t i = 0; i < size; ++i)
	{
		results = neurals[i].back_compute(loss, mode);
	}

	return results;
}

// 全连接层使用的反向传播函数
std::vector<std::vector<float>> fcLayer::backCompute(std::vector<std::vector<float>> loss)
{
	std::vector<std::vector<float>> results;
	for (size_t i = 0; i < size; ++i)
	{
		float currLoss = 0;
		for (int m = 0; m < loss.size(); ++m)
		{
			currLoss += loss[m][i];
		}

		std::vector<float> res = neurals[i].back_compute(currLoss, mode);
		results.push_back(res);
	}

	return results;
}

// 激活函数：sigmoid	---- sigmoid(y) = 1 / (1 + e^y)
void fcLayer::sigmoid()
{
	for (size_t i = 0; i < size; ++i)
	{
		float tmp = neurals.at(i).getOutput();

		tmp = 1 / (1 + expf(-tmp));
		neurals.at(i).setOutput(tmp);
	}
}

// 激活函数: ReLU ---- ReLU(y) = max(0, y);
void fcLayer::relu()
{
	for (size_t i = 1; i < size; ++i)
	{
		if (neurals.at(i).getOutput() <= 0)
		{
			neurals.at(i).setOutput(0);
		}
	}
}

// 激活函数: Tanh ---- Tanh(y) = (e^y - e^(-y)) / (e^y + e^(-y))
void fcLayer::tanh()
{
	for (size_t i = 1; i < size; ++i)
	{
		float tmp = neurals.at(i).getOutput();

		tmp = (exp(tmp) - exp(-tmp)) / (exp(tmp) + exp(-tmp));
		neurals.at(i).setOutput(tmp);
	}
}

#endif // _full_connected_layer__
