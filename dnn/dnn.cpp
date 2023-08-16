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

//// 头文件
#include "./dataCollector/dataCollector.hpp"
#include "./neural/neural.hpp"
#include "./fclayer/fcLayer.hpp"
#include "./graph/graph.hpp"

//
// dataset index: 0 ~ 62629
const size_t sizeDataset = 62630;
const int sizeTrain = 50104;
const int sizeTest = 12526;

using std::vector;

int main()
{
	std::cout << "收集训练集" << std::endl;
	dataCollector dataset("./smoke_train_data.csv");
	dataset.collectData();
	dataset.mean_normalization();


	// training parameters
	const int fc_num = 5;
	const int neural_num = 10;
	const int batch_size = 128;
	const int epoch = 10;
	const int iteration_size = (sizeDataset / batch_size) + 1;

	// random engine
	std::random_device rd;
	std::default_random_engine eng(rd());
	std::uniform_int_distribution<int> u(0, sizeTrain - 1);

	// build graph
	graph newGraph(3, 5);
	int index[batch_size];

	std::cout << "开始训练" << std::endl;
	for (int i = 0; i < 10000; ++i)
	{
		// generate batch data index
		for (int j = 0; j < batch_size; ++j)
		{
			index[j] = u(eng);
		}
		newGraph.recieveSample(dataset, index, batch_size);
		float currLoss = newGraph.forwardCompute();
		printf("After %d Iterations of training, avgLoss: %f\r", i, currLoss);
	}
	printf("\n");
	std::cout << "收集测试集" << std::endl;
	// run test
	dataCollector test("./smoke_test_data.csv");
	test.collectData();
	test.mean_normalization();
	int correct = 0;
	int cnt = 0;
	std::cout << "开始测试！" << std::endl;
	for (int i = 0; i < sizeTest; ++i)
	{
		data curr = test.get_row(i);
		float res = newGraph.testSession(curr);
		fire_label label = (res > 0.5) ? FIRE : NOFIRE;

		if (label == curr.label)
		{
			++correct;
		}
		++cnt;
		if ((i + 1) % 100 == 0)
		{
			float correctPercentage = (float)correct / (i + 1);
			correctPercentage *= 100.0f;
			printf("	Correctness : %.2f %\r", correctPercentage);
		}
	}
	float correctPercentage = (float)correct / (cnt + 1);
	correctPercentage *= 100.0f;
	printf("	Correctness : %.2f %\r", correctPercentage);
}