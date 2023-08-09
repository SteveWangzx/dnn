#ifndef _full_connected_layer__
#define _full_connected_layer__

class fcLayer
{
public:
	fcLayer() {};
	fcLayer(int num);
	float* generate_output();
	size_t getSize() { return size; }

	// activation function
	void sigmoid();
	void relu();
	void tanh();
	std::vector<neural> neurals;

private:
	size_t size;
};

fcLayer::fcLayer(const int num) {
	std::vector<neural> temp(num);
	neurals = temp;
	size = neurals.size();
}

// 激活函数：sigmoid	---- sigmoid(y) = 1 / (1 + e^y)
void fcLayer::sigmoid()
{
	for (size_t i = 0; i < size; ++i)
	{
		float tmp = neurals.at(i).getOutput();

		tmp = 1 / (1 + exp(tmp));
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
