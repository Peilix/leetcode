#pragma once
#include <vector>
#if 0
template<typename T, size_t size>
class MinStack {
public:
	/** initialize your data structure here. */
	std::array<T, size> value;
	std::array<T, size> minValue;
	size_t m_top;
	MinStack() : m_top(-1) {

	}

	void push(T x) {
		value[++m_top] = x;
		if (m_top == 0)
			minValue[m_top] = x;
		else
			minValue[m_top] = minValue[m_top - 1] > x ? x : minValue[m_top - 1];
	}

	void pop() {
		m_top--;
	}

	T top() const {
		return value[m_top];
	}

	T getMin() const {
		return minValue[m_top];
	}
};
#else
class MinStack {
public:
	/** initialize your data structure here. */
	std::vector<int> value;
	std::vector<int> minValue;
	MinStack() {

	}

	void push(int x) {
		value.push_back(x);
		if (minValue.empty()) 
			minValue.push_back(x);
		else 
			minValue.push_back(minValue.back() > x ? x : minValue.back());
	}

	void pop() {
		value.pop_back();
		minValue.pop_back();
	}

	int top() const {
		return value.back();
	}

	int getMin() const {
		return minValue.back();
	}
};
#endif


/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */

