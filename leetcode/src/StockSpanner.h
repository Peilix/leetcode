#pragma once
#include <vector>

class 	StockSpanner
{
public:
	StockSpanner() {

	}

	int next(int price) {
		stock_prices_.push_back(price);
		for (int i = stock_prices_.size() - 1; i >= 0; --i)
		{
			if (price < stock_prices_.at(i))
				return stock_prices_.size() - i - 1;
		}
		return stock_prices_.size();
	}
private:
	std::vector<int> stock_prices_;
};

/*
 *["StockSpanner","next","next","next","next","next"]
[[],[31],[41],[48],[59],[79]]

["StockSpanner","next","next","next","next","next"]
[[],[29],[91],[62],[76],[51]]
*/