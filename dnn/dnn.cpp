// 库导入
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>

// 头文件
#include "./dataCollector/dataCollector.hpp"
#include "./nueral.hpp"
#include "./fcLayer.hpp"

// Setup Init Parameters
const float LEARNING_RATE = 0.1f;

int main()
{
	dataCollector test("./smoke_detection_iot.csv");
	test.collectData();
	test.mean_normalization();
	test.peak(10);

// test of dnn
// choose one row of data
// contains 5 full-connected layers with 10 neurals each layer.
	data testRow = test.get_row(10);

	// build 5 full-connected layers.
	const int layer_size = 5;
	const int neural_size = 10;
	std::vector<fcLayer> layers;
	for (int i = 0; i < layer_size; ++i)
	{
		fcLayer temp(neural_size);
		layers.push_back(temp);
	}

	// insert data into the first layer then compute;
	for (int i = 0; i < neural_size; ++i)
	{
		layers[0].neurals[i].input_x(testRow.x, 12);
		layers[0].neurals[i].linear_compute();
	}

	layers[0].sigmoid();

	// aggregate outputs from the first layer
	float outputFirst[neural_size] = { 0 };
	for (int i = 0; i < neural_size; ++i)
	{
		outputFirst[i] = layers[0].neurals[i].getOutput();
	}

	// pass data to the next 4 fc layers
	for (int i = 1; i < layer_size; ++i)
	{
		// forward compute

		// print current outputFirst (for debug)
		std::cout << "layer " << i - 1 << " : ";
		for (int m = 0; m < neural_size; ++m)
		{
			std::cout << outputFirst[m] << " ";
		}
		std::cout << std::endl;

		for (int j = 0; j < neural_size; ++j)
		{
			layers[i].neurals[j].input_x(outputFirst, neural_size);
			layers[i].neurals[j].linear_compute();
			//outputFirst[j] = layers[i].neurals[j].getOutput();
		}

		layers[i].sigmoid();

		for (int k = 0; k < neural_size; ++k)
		{
			outputFirst[k] = layers[i].neurals[k].getOutput();
		}
	}

	// print current outputFirst (for debug)
	std::cout << "layer " << 4 << " : ";
	for (int m = 0; m < neural_size; ++m)
	{
		std::cout << outputFirst[m] << " ";
	}
	std::cout << std::endl;

	// build output layer with one single layer

	fcLayer outLayer(1);
	outLayer.neurals[0].input_x(outputFirst, 10);
	outLayer.neurals[0].linear_compute();
	outLayer.sigmoid();
	float finalOutput = outLayer.neurals[0].getOutput();

	std::cout << "DNN OUTPUT: " << finalOutput << std::endl;
}