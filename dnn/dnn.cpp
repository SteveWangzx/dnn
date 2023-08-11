//// 库导入
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <ctime>
#include <random>
#include <stdexcept>
//
//// 头文件
#include "./dataCollector/dataCollector.hpp"
#include "./nueral.hpp"
#include "./fcLayer.hpp"
#include "./graph.hpp"
//
//// dataset index: 0 ~ 62629
//const size_t sizeDataset = 62630;

//int main()
//{
//	dataCollector dataset("./smoke_detection_iot.csv");
//	dataset.collectData();
//	dataset.mean_normalization();
//
//
//	// training parameters
//	const int fc_num = 5;
//	const int neural_num = 10;
//	const int batch_size = 1000;
//	const int epoch = 10;
//	const int iteration_size = (sizeDataset / batch_size) + 1;
//
//	// random engine
//	std::random_device rd;
//	std::default_random_engine eng(rd());
//	std::uniform_int_distribution<int> u(0, sizeDataset - 1);
//
//	// build graph
//	graph newGraph(10, 10);
//	int index[batch_size];
//
//	for (int i = 0; i < 6000; ++i)
//	{
//		// generate batch data index
//		for (int j = 0; j < batch_size; ++j)
//		{
//			index[j] = u(eng);
//		}
//		newGraph.recieveSample(dataset, index, batch_size);
//		float currLoss = newGraph.forwardCompute();
//		//std::cout << "Iteration:"<< i << std::endl;
//		std::cout << "After " << i << " Iterations of training:" << std::endl;
//		std::cout << "Loss: " << currLoss << std::endl;
//		//if ((i + 1) % 1000 == 0)
//		//{
//		//	std::cout << "After " << i << " Iterations of training:" << std::endl;
//		//	std::cout << "Loss: " << fabs(currLoss) << std::endl;
//		//}
//	}
//	//newGraph.recieveSample(test, index, 1);
//
//
////// test of dnn
////// choose one row of data
////// contains 5 full-connected layers with 10 neurals each layer.
////	data testRow = test.get_row(10);
////
////	// build 5 full-connected layers.
////	const int layer_size = 5;
////	const int neural_size = 10;
////	std::vector<fcLayer> layers;
////	for (int i = 0; i < layer_size; ++i)
////	{
////		fcLayer temp(neural_size);
////		layers.push_back(temp);
////	}
////
////	// insert data into the first layer then compute;
////	for (int i = 0; i < neural_size; ++i)
////	{
////		layers[0].neurals[i].input_x(testRow.x, 12);
////		layers[0].neurals[i].linear_compute();
////	}
////
////	layers[0].sigmoid();
////
////	// aggregate outputs from the first layer
////	float outputFirst[neural_size] = { 0 };
////	for (int i = 0; i < neural_size; ++i)
////	{
////		outputFirst[i] = layers[0].neurals[i].getOutput();
////	}
////
////	// pass data to the next 4 fc layers
////	for (int i = 1; i < layer_size; ++i)
////	{
////		// forward compute
////
////		// print current outputFirst (for debug)
////		std::cout << "layer " << i - 1 << " : ";
////		for (int m = 0; m < neural_size; ++m)
////		{
////			std::cout << outputFirst[m] << " ";
////		}
////		std::cout << std::endl;
////
////		for (int j = 0; j < neural_size; ++j)
////		{
////			layers[i].neurals[j].input_x(outputFirst, neural_size);
////			layers[i].neurals[j].linear_compute();
////			//outputFirst[j] = layers[i].neurals[j].getOutput();
////		}
////
////		layers[i].sigmoid();
////
////		for (int k = 0; k < neural_size; ++k)
////		{
////			outputFirst[k] = layers[i].neurals[k].getOutput();
////		}
////	}
////
////	// print current outputFirst (for debug)
////	std::cout << "layer " << 4 << " : ";
////	for (int m = 0; m < neural_size; ++m)
////	{
////		std::cout << outputFirst[m] << " ";
////	}
////	std::cout << std::endl;
////
////	// build output layer with one single layer
////
////	fcLayer outLayer(1);
////	outLayer.neurals[0].input_x(outputFirst, 10);
////	outLayer.neurals[0].linear_compute();
////	outLayer.sigmoid();
////	float finalOutput = outLayer.neurals[0].getOutput();
////
////	std::cout << "DNN OUTPUT: " << finalOutput << std::endl;
//}

#include <vector>
#include <list>
#include <windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <ctime>
#include <random>
#include <stdexcept>
//const int X_SIZE = 12;

using std::list;
using std::vector;
enum class Actv {
	Sigmoid,
	Relu
};

class DenseLayer
{
public:
	DenseLayer(int inputsize, int outputsize, Actv actv)
	{
		this->inputszie = inputsize;
		this->num = outputsize;
		this->actv = actv;

		//分配内存
		x.resize(inputsize);
		w.resize(inputsize * outputsize);
		b.resize(outputsize);
		y.resize(outputsize);

		dx.resize(inputsize);
		dw.resize(inputsize * outputsize);
		db.resize(outputsize);
		dy.resize(outputsize);


		//初始化
		for (int n = 0; n < outputsize; ++n)
		{
			for (int i = 0; i < inputsize; ++i)
			{
				//权重初始化为0.5
				w[n * inputsize + i] = static_cast<float>(rand() - RAND_MAX / 2) / RAND_MAX;
				//权重梯度初始化为0
				dw[n * inputsize + i] = 0.0f;
			}

			//偏置初始化为0
			b[n] = 0.0f;
			//偏置梯度初始化为0
			db[n] = 0.0f;
		}
	}

	/// @brief 设置输入 
	void SetX(vector<float> x)
	{
		this->x = x;
	}
	/// @brief 获取输出 
	vector<float> GetY()
	{
		return y;
	}
	/// @brief 设置梯度输入 
	void SetDY(vector<float> dy)
	{
		this->dy = dy;
	}
	/// @brief 获取梯度输出 
	vector<float> GetDX()
	{
		return dx;
	}

	/// @brief 向前传播
	void Forward()
	{
		//y = w * x + b
		for (int n = 0; n < num; ++n)
		{
			y[n] = 0.0f;
			for (int i = 0; i < inputszie; ++i)
			{
				y[n] += x[i] * w[n * inputszie + i]; //输入 * 权重
			}
			y[n] += b[n]; //加偏置
		}

		//激活
		switch (actv)
		{
		case Actv::Relu:
		{
			for (int n = 0; n < num; ++n)
			{
				y[n] = relu(y[n]);
			}
		}break;
		case Actv::Sigmoid:
		{
			for (int n = 0; n < num; ++n)
			{
				y[n] = sigmoid(y[n]);
			}
		}break;
		}
	}

	/// @brief 向后传播
	void Backward()
	{
		//反激活
		switch (actv)
		{
		case Actv::Relu:
		{
			for (int n = 0; n < num; ++n)
			{
				dy[n] = relu_gd(y[n]) * dy[n];
			}
		}break;
		case Actv::Sigmoid:
		{
			for (int n = 0; n < num; ++n)
			{
				dy[n] = sigmoid_gd(y[n]) * dy[n];
			}
		}break;
		}

		//计算偏置梯度 (y = w * x + b  b求偏导: db = 1.0)
		for (int n = 0; n < num; ++n)
		{
			db[n] += 1.0f * dy[n];
		}

		//计算权重梯度 (y = w * x + b  w求偏导: dw = x)
		for (int n = 0; n < num; ++n)
		{
			for (int i = 0; i < inputszie; ++i)
			{
				dw[n * inputszie + i] += x[i] * dy[n];
			}
		}

		//计算输入梯度 (y = w * x + b  x求偏导: dx = w)  用于前一层反向传播
		for (int i = 0; i < inputszie; ++i)
		{
			dx[i] = 0.0f; //dx无需累积，每次计算前清零
			for (int n = 0; n < num; ++n)
			{
				dx[i] += w[n * inputszie + i] * dy[n];
			}
		}
	}

	/// @brief 更新参数
	void Update()
	{
		static const float LR = 0.03f; //学习率
		static const float Momenteum = 0.9f; //动量(不是必须的)

		//更新权重
		for (int idx = 0; idx < inputszie * num; ++idx)
		{
			w[idx] += dw[idx] * LR;
		}
		//更新偏置
		for (int idx = 0; idx < num; ++idx)
		{
			b[idx] += db[idx] * LR;
		}


		//动量策略（如果未设置动量，直接清零即可）
		for (int idx = 0; idx < inputszie * num; ++idx)
		{
			dw[idx] *= Momenteum;
		}
		for (int idx = 0; idx < num; ++idx)
		{
			db[idx] *= Momenteum;
		}
	}
private:
	float sigmoid(float x)
	{
		return 1.0f / (1.0f + expf(-x));
	}
	float sigmoid_gd(float y)
	{
		return y * (1.0f - y);
	}
	float relu(float x)
	{
		if (x > 0) { return x; }
		else { return 0.0f; }
	}
	float relu_gd(float y /*严格来说，应该输入x*/)
	{
		if (y > 0) { return 1.0f; }
		else { return 0.0f; }
	}



private:
	int		inputszie; //输入数据长度
	int		num;//神经元个数（输出数据长度）
	Actv	actv;

	vector<float> x;
	vector<float> w;
	vector<float> b;
	vector<float> y;

	vector<float> dx;
	vector<float> dw;
	vector<float> db;
	vector<float> dy;
};


int main(int argc, char** argv)
{
	struct Sample
	{
		vector<float> x;
		vector<float> t;
	};
	int batch_size = 8;
	//建立神经计算图
	vector<DenseLayer> layers;
	layers.push_back(DenseLayer(12, 8, Actv::Relu));
	layers.push_back(DenseLayer(8, 3, Actv::Relu));
	layers.push_back(DenseLayer(3, 1, Actv::Sigmoid));

	//创建简单的样本
	dataCollector dataset("./smoke_detection_iot.csv");
	dataset.collectData();
	dataset.mean_normalization();

	vector<Sample> samples;
	for (int i = 0; i < 62630; ++i)
	{
		Sample sample;
		data tmp = dataset.get_row(i);
		for (int j = 0; j < 12; ++j)
		{
			sample.x.push_back(tmp.x[j]);
		}
		sample.t.push_back((float)tmp.label); //假定各值相加大于0为报警，小于0为不报警
		samples.push_back(sample);
	}

	list<float> losslist;

	//训练
	for (int i = 0; i < 10000; ++i)
	{
		for (int b = 0; b < batch_size; ++b)
		{
			long long rn = rand() * rand() % 10000;
			Sample& sample = samples[rn];

			//向前传播
			//////////////////////////////////////////////////////////////////////////

			layers[0].SetX(sample.x);
			layers[0].Forward();

			layers[1].SetX(layers[0].GetY());
			layers[1].Forward();

			layers[2].SetX(layers[1].GetY());
			layers[2].Forward();


			//根据样本计算梯度
			//////////////////////////////////////////////////////////////////////////

			vector<float> y = layers[2].GetY();
			vector<float> loss(1);
			loss[0] = sample.t[0] - y[0];

			{//取最近100次的loss平均值，观察loss的变化
				losslist.push_back(abs(loss[0]));
				if (losslist.size() > 100) { losslist.pop_front(); }
				float tloss = 0;
				for (float l : losslist) { tloss += l; }
				tloss /= losslist.size();
				printf("avgloss:%f\r", tloss);
				::Sleep(1);
			}

			//反向传播
			//////////////////////////////////////////////////////////////////////////

			layers[2].SetDY(loss);
			layers[2].Backward();

			layers[1].SetDY(layers[2].GetDX());
			layers[1].Backward();

			layers[0].SetDY(layers[1].GetDX());
			layers[0].Backward();
		}
		//更新所有层的权重
		//////////////////////////////////////////////////////////////////////////
		layers[0].Update();
		layers[1].Update();
		layers[2].Update();
	}

	return 0;
}