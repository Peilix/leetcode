#pragma once
#include <chrono>

class Timer
{
public:
	Timer();
	~Timer();
	void stop();
private:
	std::chrono::steady_clock::time_point start_point_;
};

