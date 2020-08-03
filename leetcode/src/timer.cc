#include "timer.h"

#include <iostream>

Timer::Timer()
{
	start_point_ = std::chrono::high_resolution_clock::now();
}

Timer::~Timer()
{
	stop();
}

void Timer::stop()
{
	auto  tick = std::chrono::high_resolution_clock::now();
	auto start = std::chrono::time_point_cast<std::chrono::microseconds>(start_point_).time_since_epoch();
	auto end = std::chrono::time_point_cast<std::chrono::microseconds>(tick).time_since_epoch();
	auto duration = end - start;
	//std::cout << duration << ""
}
