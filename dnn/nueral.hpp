#ifndef _neural_hpp__
#define _nueral_hpp__

/********************
* Header for nueral *
********************/

const float INIT_WEIGHT = 0.5f;
const float INIT_BIAS = 0.1f;

class neural
{
public:
	neural() : weights(INIT_WEIGHT), bias(INIT_BIAS) {};
	neural(float x[], size_t size) : x_size(size), input(x), weights(INIT_WEIGHT), bias(INIT_BIAS) {};
	void input_x(float x[], size_t size);
	void linear_compute();
	void print_nueral();
	float getOutput() { return output; }
	void setOutput(float updateOutput) { output = updateOutput; }
	void back_compute(float loss);

protected:
	float weights;
	float bias;

private:
	float *input;
	size_t x_size;
	float output = 0;
};

void neural::input_x(float x[], size_t size)
{
	input = x;
	x_size = size;
}

// linear_compute()
// @brief: linear forward compute ---- y = w * x + b 
// @param:
void neural::linear_compute()
{
	output = 0;
	for (int i = 0; i < x_size; ++i)
	{
		output += input[i] * weights + bias;
	}
	std::cout << "linear output: " << output << " " << std::endl;
}

void neural::print_nueral()
{
	std::cout << "Nueral Input :";
	for (int i = 0; i < x_size; ++i)
	{
		std::cout << input[i] << " ";
	}
	std::cout << std::endl;
}

#endif // _neural_hpp__
