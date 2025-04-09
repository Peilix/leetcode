#pragma once

#include <vector>
#include <queue>
#include <string>
#include <map>
#include <set>
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

class WordDictionary {
	struct TrieNode {
		std::map<char, TrieNode *> children;
		bool is_end = false;
	};
	TrieNode *root;

    public:
	/** Initialize your data structure here. */
	WordDictionary()
	{
		root = new TrieNode();
	}

	/** Adds a word into the data structure. */
	void addWord(std::string word)
	{
		TrieNode *curr = root;
		for (auto letter : word) {
			if (curr->children.find(letter) ==
			    curr->children.end()) {
				curr->children[letter] = new TrieNode();
			}
			curr = curr->children[letter];
		}
		curr->is_end = true;
	}

	/** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
	bool search(std::string word)
	{
		return searchRecursive(root, word, 0);
	}

    private:
	bool searchRecursive(TrieNode *curr, std::string &word, size_t index)
	{
		if (index == word.size())
			return curr->is_end;
		char letter = word[index];
		if (letter == '.') {
			for (const auto &e : curr->children)
				if (searchRecursive(e.second, word, index + 1))
					return true;
		} else {
			if (curr->children.find(letter) == curr->children.end())
				return false;
			return searchRecursive(curr->children[letter], word,
					       index + 1);
		}
		return false;
	}
};

/**
 * Your WordDictionary object will be instantiated and called as such:
 * WordDictionary* obj = new WordDictionary();
 * obj->addWord(word);
 * bool param_2 = obj->search(word);
<<<<<<< HEAD
 */

class TaskManager {
    public:
	TaskManager(std::vector<std::vector<int> > &tasks)
	{
	}

	void add(int userId, int taskId, int priority)
	{
	}

	void edit(int taskId, int newPriority)
	{
	}

	void rmv(int taskId)
	{
	}

	int execTop()
	{
		return 0;
	}

    private:
};

/**
 * Your TaskManager object will be instantiated and called as such:
 * TaskManager* obj = new TaskManager(tasks);
 * obj->add(userId,taskId,priority);
 * obj->edit(taskId,newPriority);
 * obj->rmv(taskId);
 * int param_4 = obj->execTop();
 */

class NumberContainers {
    public:
	NumberContainers()
	{
	}

	void change(int index, int number)
	{
		if (index_to_number_.contains(index)) {
			auto previous_number = index_to_number_[index];
			number_to_indices_[previous_number].erase(index);
			if (number_to_indices_[previous_number].empty()) {
				number_to_indices_.erase(previous_number);
			}
		}
		index_to_number_[index] = number;
		number_to_indices_[number].insert(index);
	}

	int find(int number)
	{
		if (number_to_indices_.contains(number)) {
			return *number_to_indices_[number].begin();
		}
		return -1;
	}

    private:
	std::unordered_map<int, int> index_to_number_;
	std::unordered_map<int, std::set<int> > number_to_indices_;
};

/**
 * Your NumberContainers object will be instantiated and called as such:
 * NumberContainers* obj = new NumberContainers();
 * obj->change(index,number);
 * int param_2 = obj->find(number);
 */

class LockingTree {
    private:
	std::unordered_map<int, std::vector<int> > children_;
	std::unordered_map<int, int > parent_;
	std::vector<std::pair<bool,int>> locked_;
    public:
	LockingTree(std::vector<int> &parent)
	{
		// parent[0] == -1
		parent_[0] = -1;
		for (int i = 1; i < parent.size(); i++) {
			parent_[i] = parent[i];
			children_[parent[i]].push_back(i);
		}
		locked_.resize(parent.size(), { false, -1 });
	}

	// Locks the given node for the given user and prevents other users from locking the same node. You may only lock a node using this function if the node is unlocked.
	bool lock(int num, int user)
	{
		if (locked_[num].first) {
			return false;
		}
		locked_[num] = { true, user };
		return true;
	}

	// Unlocks the given node for the given user. You may only unlock a node using this function if it is currently locked by the same user.
	bool unlock(int num, int user)
	{
		if (locked_[num].first and locked_[num].second == user) {
			locked_[num] = { false, -1 };
			return true;
		}
		return false;
	}

	// Locks the given node for the given user and unlocks all of its descendants regardless of who locked it. You may only upgrade a node if all 3 conditions are true:\
    // The node is unlocked,
	// It has at least one locked descendant(by any user),
	// and It does not have any locked ancestors.
	bool upgrade(int num, int user)
	{
		// The node is unlocked
		if (locked_[num].first) {
			return false;
		}
		// It does not have any locked ancestors
		auto parent = parent_[num];
		while (-1 != parent and not locked_[parent].first) {
			parent = parent_[parent];
		}
		if (-1 != parent) {
			return false;
		}
		//It has at least one locked descendant (by any user)
		std::queue<int> q(children_[num].begin(), children_[num].end());
		std::vector<int> to_unlock;
		for (; not q.empty();) {
			auto n = q.front();
			q.pop();
			if (locked_[n].first) {
				to_unlock.push_back(n);
			}
			for (auto child : children_[n]) {
				q.push(child);
			}
		}
		if (to_unlock.empty()) {
			return false;
		}
		
		// Locks the given node for the given user and unlocks all of its descendants regardless of who locked it.
		locked_[num] = { true, user };
		for (auto descendant : to_unlock) {
			locked_[descendant] = { false, -1 };
		}
		return true;
	}
};

/**
 * Your LockingTree object will be instantiated and called as such:
 * LockingTree* obj = new LockingTree(parent);
 * bool param_1 = obj->lock(num,user);
 * bool param_2 = obj->unlock(num,user);
 * bool param_3 = obj->upgrade(num,user);
=======
>>>>>>> 814e0a7adb23eebc5e1c35397dc0cc60509111e7
 */