#ifndef _neural_hpp__
#define _nueral_hpp__

/********************
* Header for nueral *
********************/

const float LEARNING_RATE = 0.1f;
const float INIT_WEIGHT = 0.5f;
const float INIT_BIAS = 0.1f;

class neural
{
public:
	neural() : bias(INIT_BIAS), input(0), x_size(0) {};
	neural(float x[], const size_t size) : x_size(size),  bias(INIT_BIAS) 
	{
		for (size_t i = 0; i < x_size; ++i)
		{
			input.push_back(x[i]);
		}
		if (weights.empty())
		{
			for (size_t i = 0; i < size; ++i)
			{
				weights.push_back(INIT_WEIGHT);
			}
		}
	};
	void input_x(float x[], size_t size);
	void input_x(std::vector<float>& x, size_t size);
	void linear_compute();
	void print_neural();
	float getOutput() { return output; }
	void setOutput(float updateOutput) { output = updateOutput; }
	std::vector<float> back_compute(float loss);
	std::vector<float> back_compute(std::vector<float> loss);
	size_t getSize() { return x_size; }
	std::vector<float> weights;
	float bias;

private:
	std::vector<float> input;
	size_t x_size;
	float output = 0;
	float linear_output = 0;
};

// input_x(float x[], size_t size)
// @brief: pass input to a default constructed nueral
void neural::input_x(float x[], size_t size)
{
	input.clear();
	x_size = size;
	for (size_t i = 0; i < size; ++i)
	{
		input.push_back(x[i]);
	}
	if (weights.empty())
	{
		for (size_t i = 0; i < size; ++i)
		{
				weights.push_back(INIT_WEIGHT);
		}
	}
}

// input_x(std::vector<float>& x, size_t size) 重载
void neural::input_x(std::vector<float>& x, size_t size)
{
	input.clear();
	for (size_t i = 0; i < size; ++i)
	{

		input.push_back(x[i]);

	}
	if (weights.empty())
	{
		for (size_t i = 0; i < size; ++i)
		{
			weights.push_back(INIT_WEIGHT);
		}
	}
	x_size = size;
}

/******************************************************
* linear_compute()                                    *
* @brief: linear forward compute ---- y = w * x + b   *
* @param:                                             *
******************************************************/
void neural::linear_compute()
{
	output = 0;
	for (int i = 0; i < x_size; ++i)
	{
		//std::cout << "weights" << weights[i] << std::endl;

		output += input[i] * weights[i];

		
	}
	//std::cout << "bias" << bias << std::endl;
	output += bias;
	linear_output = output;
}

// Back Propagation ---- sigmoid(y) = 1 / (1 + e^-y)
//					---- y = w * x + b;
//					---- 复合函数求导: f(g(x)) = f'(g) * g'(x);
//					---- 求导: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
//					---- 偏导：
//					---- d(sigmoid) / d(w) = sigmoid' * (w * x + b)'
//					----				   = sigmoid'(w * x + b) * x 
//					----                   = x * (sigmoid(w *x + b) * (1 - sigmoid(w * x + b))) 
//					---- d(sigmoid) / d(b) = sigmoid' * (w * x + b)'
//					----				   = sigmoid'(w * x + b) * 1
//					----				   = sigmoid(w * x + b) * (1 - sigmoid(w * x + b))
std::vector<float> neural::back_compute(float loss)
{
	float sigmoid_wxb = (1 / (1 + exp(-linear_output))) * (1 - (1 / (1 + exp(-linear_output))));
	std::vector<float> dx;

	// dx = w * loss
	for (int i = 0; i < x_size; ++i)
	{
		dx.push_back(weights[i] * loss);
	}

	// dw ---- update on weights
	for (int i = 0; i < x_size; ++i)
	{
		weights[i] += input[i] * sigmoid_wxb * loss * LEARNING_RATE;
	}

	// db ---- update on bias
	bias += sigmoid_wxb * loss * LEARNING_RATE;

	return dx;
}

/**********************************
* print_neural()                  *
* @brief: print input of a neural *
**********************************/
void neural::print_neural()
{
	std::cout << "Nueral Input :";
	for (int i = 0; i < x_size; ++i)
	{
		std::cout << input[i] << " ";
	}
	std::cout << std::endl;
}

#endif // _neural_hpp__
