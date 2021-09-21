#include "Timer.h"

Timer::Timer()
{
	start = std::chrono::high_resolution_clock::now();
}
Timer::~Timer()
{
	Stop();
}
void Timer::re_start()
{
	auto lambda = [this]() {
		Stop();
		start = std::chrono::high_resolution_clock::now(); 
	};
	lambda();
}
void Timer::Stop()
{
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;

	float ms = duration.count() * 1000.0f;
	std::cout << "Timer took " << ms << "ms" << std::endl;
}