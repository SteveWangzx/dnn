#ifndef _data_collector_hpp__
#define _data_collector_hpp__

/***************************
* Header for dataCollector *
***************************/

// dataset - smoke detection 
// index
// 0. Temperature - Temperature of Surroundings.Measured in Celsius                         - float64
// 1. Humidity - The air humidity during the experiment.                                    - float64
// 2. TVOC - Total Volatile Organic Compounds.Measured in ppb(parts per billion)            - int64
// 3. eCo2 - CO2 equivalent concentration.Measured in ppm(parts per million)                - int64
// 4. Raw H2 - The amount of Raw Hydrogen present in the surroundings.                      - int64
// 5. Raw Ethanol - The amount of Raw Ethanol present in the surroundings.                  - int64
// 6. Pressure - Air pressure.Measured in hPa                                               - float64
// 7. PM1.0 - Paticulate matter of diameter less than 1.0 micrometer .                      - float64
// 8. PM2.5 - Paticulate matter of diameter less than 2.5 micrometer.                      - float64
// 9. NC0.5 - Concentration of particulate matter of diameter less than 0.5 micrometers.   - float64
// 10. NC1.0 - Concentration of particulate matter of diameter less than 1.0 micrometers.   - float64
// 11. NC2.5 - Concentration of particulate matter of diameter less than 2.5 micrometers.   - float64
// Label - Fire Alarm - (Reality)If fire was present then value is 1 else it is 0.              - int64

const int X_SIZE = 12;

enum fire_label {
	NOFIRE,
	FIRE
};

struct data
{
	//float temperature;              // 温度
	//float humidity;					// 湿度
	//int TVOC;					
	//int eCo2;
	//int rawH2;
	//int rawEthanol;
	//float pressure;
	//float pm1_0;
	//float pm2_5;
	//float nc0_5;
	//float nc1_0;
	//float nc2_5;
	float x[12];
	fire_label label;
};

struct normStructPattern
{
	float min = 0;
	float max = 0;
	float mean = 0;
};
typedef normStructPattern normStruct;

const normStruct x_norm[X_SIZE] = {
	{	// temperature
		-22.0100f,
		59.9300f,
		15.9704f
	},
	{	// Humidity
		10.7400f,
		75.2000f,
		48.5394f,
	},
	{	// TVOC
		0.0f,
		60000.0f,
		1942.0575f
	},
	{	// eCo2
		400.0000f,
		60000.0f,
		670.0210f
	},
	{	// Raw H2
		10668.0f,
		13803.0f,
		12942.4539f
	},
	{	// Raw Ethanol
		15317.0f,
		21410.0f,
		19754.2579f
	},
	{	// Pressure
		930.8520f,
		939.8610f,
		938.6276f
	},
	{	// PM1.0
		0.0f,
		14333.69f,
		100.5943f
	},
	{	// PM2.5
		0.0f,
		45432.26f,
		184.4677f
	},
	{	// NC0.5
		0.0f,
		61482.03f,
		491.4636f
	},
	{	// NC1.0
		0.0f,
		51914.68f,
		203.5864f
	},
	{	// NC2.5
		0.0f,
		30026.4380f,
		80.0490f
	},
};

class dataCollector
{
public:
	// constructor
	dataCollector() {};
	dataCollector(const char* path);

	// member function
	void collectData();
	void peak(const int row);
	void peak_norm();
	//void generate_norm();
	void mean_normalization();

private:
	void dataHandler(std::vector<std::string>& words);

	std::string pathDataset;
	std::vector<data> samples;
	//std::vector<std::vector<float>> norm_samples;
};


// dataCollector(const char* path) - 构造函数
// @brief: 
// @param: csv文件路径
dataCollector::dataCollector(const char* path)
{
	pathDataset = path;
}

// void dataCollector::collectData()
// @brief: 读取csv文件
// @param: 
void dataCollector::collectData()
{
	std::ifstream csvFile(pathDataset, std::ios::in);
	std::string line, word;
	std::istringstream sin;
	std::vector<std::string> words;

	if (!csvFile.is_open())
	{
		std::cout << "File Not Found!" << std::endl;
		exit(1);
	}
	std::cout << "File Open Successfully!" << std::endl;

	// 跳过标题
	std::getline(csvFile, line);
	// 读取数据
	while (std::getline(csvFile, line))
	{
		if (line.length() == 0)
		{
			continue;
		}

		sin.clear();
		sin.str(line);
		words.clear();

		while (std::getline(sin, word, ','))
		{
			words.push_back(word);
		}
		// 处理数据
		dataHandler(words);
	}
	std::cout << "Collect Data Successfully!" << std::endl;
	csvFile.close();
}

// void dataCollector::dataHandler
// @brief: 转化csv文本数据到struct data
// @param: vector words ---- csv文件某一行的文本内容
void dataCollector::dataHandler(std::vector<std::string> &words)
{
	data temp;
	int idx = 2;
	for (int i = 0; i < X_SIZE; ++i)
	{
		temp.x[i] = std::stof(words[idx]);
		++idx;
	}
	temp.label = std::stoi(words[15]) == 1 ? FIRE : NOFIRE;

	samples.push_back(temp);
}

// peak(int row)
// @brief: 输出任意行的数据及标签
// @param: row col
void dataCollector::peak(int row)
{
	auto begin = samples.begin();
	begin += row;
	std::cout << "X: ";
	for (int i = 0; i < X_SIZE; ++i)
	{
		std::cout << (*begin).x[i] << "  ";
	}
	std::cout << std::endl;
	std::cout << "Y: " << (*begin).label << std::endl;
}

void dataCollector::peak_norm() 
{
	std::cout << "min: " << x_norm[0].min << "  	max: " << x_norm[0].max
		<< "	mean: " << x_norm[0].mean << std::endl;
}

// @brief: 对数据集进行mean归一化处理	---- x(norm) = x - X(mean) / X(max) - X(min)
//									---- 区间：	[-1, 1]
void dataCollector::mean_normalization()
{
	int size = samples.size();
	auto begin = samples.begin();

	for (int i = 0; i < X_SIZE; ++i)
	{
		float min = x_norm[i].min;
		float max = x_norm[i].max;
		float mean = x_norm[i].mean;

		for (int j = 0; j < size; ++j)
		{
			auto samples_idx = begin + j;
			float input = (*samples_idx).x[i];
			float output = (input - mean) / (max - min);
			(*samples_idx).x[i] = output;
		}
	}
}

#endif