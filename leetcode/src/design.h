#pragma once
#include <vector>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
class WordFilter {
    public:
	WordFilter(std::vector<std::string> &words)
	{
	}

	int f(std::string prefix, std::string suffix)
	{
		return 0;
	}
};

/**
 * Your WordFilter object will be instantiated and called as such:
 * WordFilter* obj = new WordFilter(words);
 * int param_1 = obj->f(prefix,suffix);
 */

class StockPrice {
    public:
	StockPrice()
	{
	}

	void update(int timestamp, int price)
	{
		if (data_.find(timestamp) != data_.end()) {
			delayed_delete_price_max_heap_[data_[timestamp]]++;
			delayed_delete_price_min_heap_[data_[timestamp]]++;
			while (!max_heap_.empty() &&
			       delayed_delete_price_max_heap_.find(
				       max_heap_.top()) !=
				       delayed_delete_price_max_heap_.end() &&
			       0 < delayed_delete_price_max_heap_
					       [max_heap_.top()]) {
				delayed_delete_price_max_heap_[max_heap_.top()]--;
				max_heap_.pop();
			}

			while (!min_heap_.empty() &&
			       delayed_delete_price_min_heap_.find(
				       min_heap_.top()) !=
				       delayed_delete_price_min_heap_.end() &&
			       0 < delayed_delete_price_min_heap_
					       [min_heap_.top()]) {
				delayed_delete_price_min_heap_[min_heap_.top()]--;
				min_heap_.pop();
			}
		}

		data_[timestamp] = price;
		max_heap_.push(price);
		min_heap_.push(price);
	}

	int current()
	{
		return data_.crbegin()->second;
	}

	int maximum()
	{
		return max_heap_.top();
	}

	int minimum()
	{
		return min_heap_.top();
	}

    private:
	std::map<int, int> data_;
	std::priority_queue<int> max_heap_;
	std::priority_queue<int, std::vector<int>, std::greater<int> > min_heap_;
	std::unordered_map<int, int> delayed_delete_price_max_heap_;
	std::unordered_map<int, int> delayed_delete_price_min_heap_;
};

/**
 * Your StockPrice object will be instantiated and called as such:
 * StockPrice* obj = new StockPrice();
 * obj->update(timestamp,price);
 * int param_2 = obj->current();
 * int param_3 = obj->maximum();
 * int param_4 = obj->minimum();
 */

class LRUCache {
    public:
	LRUCache(int capacity) : capacity_(capacity)
	{
	}

	int get(int key)
	{
		if (hash_map_.find(key) == hash_map_.end())
			return -1;
		list_.erase(hash_map_.at(key).second);
		list_.push_front(key);
		hash_map_[key].second = list_.begin();
		return hash_map_[key].first;
	}

	void put(int key, int value)
	{
		if (hash_map_.find(key) != hash_map_.end()) {
			list_.erase(hash_map_.at(key).second);
		} else if (list_.size() == capacity_) {
			hash_map_.erase(list_.back());
			list_.pop_back();
		}
		list_.push_front(key);
		hash_map_[key] = std::make_pair(value, list_.begin());
	}

    private:
	std::list<int> list_;
	std::unordered_map<int, std::pair<int, std::list<int>::iterator> >
		hash_map_;
	size_t capacity_;
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */

class Twitter {
    public:
	Twitter()
	{
	}

	void postTweet(int userId, int tweetId)
	{
		tweets_.push_front(std::make_pair(userId, tweetId));
	}

	std::vector<int> getNewsFeed(int userId)
	{
		std::vector<int> ret;
		for (const auto &pair : tweets_) {
			if (pair.first == userId ||
			    follow_table_[userId].find(pair.first) !=
				    follow_table_[userId].end())
				ret.push_back(pair.second);
			if (ret.size() == capacity_)
				break;
		}
		return ret;
	}

	void follow(int followerId, int followeeId)
	{
		follow_table_[followerId].insert(followeeId);
	}

	void unfollow(int followerId, int followeeId)
	{
		follow_table_[followerId].erase(followeeId);
	}

    private:
	std::unordered_map<int, std::set<int> > follow_table_;
	std::list<std::pair<int, int> > tweets_;
	const size_t capacity_ = 10;
};

/**
 * Your Twitter object will be instantiated and called as such:
 * Twitter* obj = new Twitter();
 * obj->postTweet(userId,tweetId);
 * vector<int> param_2 = obj->getNewsFeed(userId);
 * obj->follow(followerId,followeeId);
 * obj->unfollow(followerId,followeeId);
 */

class RandomizedSet {
    public:
	RandomizedSet()
	{
	}
	bool insert(int val)
	{
		if (indices_.contains(val))
			return false;
		indices_[val] = numbers_.size();
		numbers_.push_back(val);
		return true;
	}
	bool remove(int val)
	{
		if (!indices_.contains(val))
			return false;

		int last = numbers_.back();
		indices_[last] = indices_[val];
		numbers_[indices_[val]] = last;
		numbers_.pop_back();
		indices_.erase(val);
		return true;
	}
	int getRandom()
	{
		return numbers_[rand() % numbers_.size()];
	}

    private:
	std::vector<int> numbers_;
	std::unordered_map<int, size_t> indices_;
};

/**
 * Your RandomizedSet object will be instantiated and called as such:
 * RandomizedSet* obj = new RandomizedSet();
 * bool param_1 = obj->insert(val);
 * bool param_2 = obj->remove(val);
 * int param_3 = obj->getRandom();
 */
