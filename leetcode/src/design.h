#pragma once
#include <string>
#include <unordered_map>
class Logger {
private:
	std::unordered_map<std::string, int> umap;
public:
	Logger() {}
	bool shouldPrintMessage(int timestamp, std::string message) {
		if (0 == umap.count(message) || 10 <= timestamp - umap[message])
		{
			umap[message] = timestamp;
			return true;
		}
		return false;
	}
};

class MyHashSet {
public:
    /** Initialize your data structure here. */
    MyHashSet() {
		grid.resize(1001);
		for (size_t i = 0; i < grid.size(); i++)
		{
			grid[i] = std::vector<bool>(1000, false);
		}
    }

    void add(int key) {
		grid[key / 1000][key % 1000] = true;
    }

    void remove(int key) {
		grid[key / 1000][key % 1000] = false;
    }

    /** Returns true if this set contains the specified element */
    bool contains(int key) {
		return grid[key / 1000][key % 1000];
    }
private:
	std::vector<std::vector<bool>> grid;
};

/**
 * Your MyHashSet object will be instantiated and called as such:
 * MyHashSet* obj = new MyHashSet();
 * obj->add(key);
 * obj->remove(key);
 * bool param_3 = obj->contains(key);
 */

class MyHashMap {
public:
    /** Initialize your data structure here. */
    MyHashMap() {

    }

    /** value will always be non-negative. */
    void put(int key, int value) {

    }

    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    int get(int key) {

    }

    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    void remove(int key) {

    }
};

/**
 * Your MyHashMap object will be instantiated and called as such:
 * MyHashMap* obj = new MyHashMap();
 * obj->put(key,value);
 * int param_2 = obj->get(key);
 * obj->remove(key);
 */