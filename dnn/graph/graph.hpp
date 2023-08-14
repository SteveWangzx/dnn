#ifndef _graph_hpp__
#define _graph_hpp__

/**********************************************
* Header for Neural Network Graph (Structure) *
**********************************************/

#include "../neural/neural.hpp"

class graph
{
public:
	graph(): layer_size(0), neural_size(0) {};
	graph(const int num_fc_layers, const int num_fc_nuerals);
	void recieveSample(dataCollector &sample, const int index[],
		const int batchSize);
	float forwardCompute();
	void backcompute(float loss);
	void printForwardParam(std::vector<float> &currOutput, const int currlayer);

	std::vector<fcLayer> fcLayers;

private:
	float loss();

private:
	std::vector<data> batchSamples;
	fcLayer outlayer;
	float outputDnn = 0;
	float lossSum = 0;
	float miniLossSum = 0;
	int layer_size = 0;
	int neural_size = 0;
};

// constructor -- 
graph::graph(const int num_fc_layers, const int num_fc_nuerals)
{
	layer_size = num_fc_layers;
	neural_size = num_fc_nuerals;

	fcLayer outTemp(1, Actv::SIGMOID);
	outlayer = outTemp;

	for (size_t i = 0; i < layer_size; ++i)
	{
		fcLayers.push_back(fcLayer(neural_size, Actv::RELU));
	}
}

void graph::printForwardParam(std::vector<float>& currOutput, const int currlayer)
{
	std::cout << "layer " << currlayer << " : ";
	for (int m = 0; m < neural_size; ++m)
	{
		std::cout << currOutput[m] << " ";
	}
	std::cout << std::endl;

}

// recieveSample() -- Recieve input from main
void graph::recieveSample(dataCollector& samples, const int index[], const int batchSize)
{
	batchSamples.clear();  // clear vector first

	for (int i = 0; i < batchSize; ++i)
	{
		int currIdx = index[i];
		data tmp = samples.get_row(currIdx);
		
		batchSamples.push_back(tmp);
	}
}

// forwardCompute() --
float graph::forwardCompute()
{
	size_t size = batchSamples.size();
	int cnt = 1;
	int order = 0;
	int sizeMiniBatch = 128;
	std::vector<float> currOutput;
	float currLoss = 0;
	lossSum = 0;

	// 遍历每条数据 in the batch
	for (size_t i = 0; i < size; ++i)
	{
			data tmp = batchSamples[i];
			currOutput.clear();
			// 输入并计算第一层
			for (size_t j = 0; j < neural_size; ++j)
			{
				fcLayers[0].neurals[j].input_x(tmp.x, 12);
				fcLayers[0].neurals[j].linear_compute();
			}
 			fcLayers[0].relu();
			//std::cout << "第一层: ";
			for (size_t j = 0; j < neural_size; ++j)
			{
				//std::cout << fcLayers[0].neurals[j].getOutput() << " ";
				currOutput.push_back(fcLayers[0].neurals[j].getOutput());
			}
			//std::cout << std::endl;

			// 计算后续层
			for (size_t j = 1; j < layer_size; ++j)
			{

				for (size_t k = 0; k < neural_size; ++k)
				{
					fcLayers[j].neurals[k].input_x(currOutput, neural_size);
					fcLayers[j].neurals[k].linear_compute();
				}
				fcLayers[j].relu();

				//std::cout << "第" << j << "层： ";
				for (size_t l = 0; l < neural_size; ++l)
				{
					//std::cout << fcLayers[j].neurals[l].getOutput() << " ";
					currOutput[l] = fcLayers[j].neurals[l].getOutput();
				}
				//std::cout << std::endl;
			}
			outlayer.neurals[0].input_x(currOutput, neural_size);
			outlayer.neurals[0].linear_compute();
			outlayer.sigmoid();

			//std::cout << "Dnn output: " << outlayer.neurals[0].getOutput() << std::endl;
			outputDnn = outlayer.neurals[0].getOutput();
			currLoss = tmp.label - outputDnn;
			lossSum += fabs(currLoss);
			//std::cout << "Loss: " << lossSum / (i + 1) << std::endl;


			backcompute(currLoss);

	}

	// print out first layer first neurals' weights and bias
	std::cout << "Weights:" << std::endl;
	for (int m = 0; m < fcLayers[1].neurals[0].getSize(); ++m)
	{
		std::cout << fcLayers[1].neurals[0].weights[m] << " ";
	}
	std::cout << std::endl;
	std::cout << "Bias:" << fcLayers[1].neurals[0].bias << std::endl;

	// return batch average loss
	return lossSum / size;
}

void graph::backcompute(float loss)
{
	std::vector<std::vector<float>> layerLoss;

	// compute for outputlayer
	layerLoss.push_back(outlayer.backCompute(loss));

	for (int i = layer_size - 1; i >= 0; --i)
	{
		layerLoss = fcLayers[i].backCompute(layerLoss);	
	}
}

//float graph::loss()
//{
//
//}

#endif // !_graph_hpp__
