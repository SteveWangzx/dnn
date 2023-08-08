// 库导入
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

// 头文件
#include "./dateCollector/dataCollector.hpp"

int main()
{
	dataCollector test("./smoke_detection_iot.csv");
	test.collectData();
	test.peak(10);
	test.mean_normalization();
	test.peak(10);
}