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

class Trie {
public:
    /** Initialize your data structure here. */
    Trie() {

    }

    /** Inserts a word into the trie. */
    void insert(std::string word) {

    }

    /** Returns if the word is in the trie. */
    bool search(std::string word) {

    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(std::string prefix) {

    }

private:
    std::vector<Trie> links;
    int R = 26;
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */

class WordDictionary {
public:
    /** Initialize your data structure here. */
    WordDictionary() {

    }

    /** Adds a word into the data structure. */
    void addWord(std::string word) {

    }

    /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
    bool search(std::string word) {

    }
};

/**
 * Your WordDictionary object will be instantiated and called as such:
 * WordDictionary* obj = new WordDictionary();
 * obj->addWord(word);
 * bool param_2 = obj->search(word);
 */

class CombinationIterator {
#if true
public:
    CombinationIterator(std::string characters, int combinationLength)
        : length_(combinationLength), characters_(characters)
    {
        bitmask_ = (1 << characters.length()) - 1;
    }

    std::string next() {
        std::string res;
        auto bitCount = [](int n) {
            unsigned bitcount = 0;
            for (; n != 0; bitcount++)
                n &= n - 1;
            return bitcount;
        };
        while (bitCount(--bitmask_) != length_)
            ;
        for (int i = characters_.size() - 1; i >= 0; i--)
            if (((1 << i) & bitmask_) != 0)
                res.push_back(characters_.at(characters_.size() - i - 1));
        return res;
    }

    bool hasNext() {
        return (1 << length_) - 1 < bitmask_;
    }

private:
    std::string characters_;
    int length_;
    int bitmask_;
#else
public:
    CombinationIterator(std::string characters, int combinationLength) {
        int n = characters.length();
        if (0 == n || 0 == combinationLength)
            combinations_.push_back(std::string());
        else if (n == combinationLength)
            combinations_.push_back(characters);
        else
        {
            std::string characters_except_first(characters.begin() + 1, characters.end());
            std::vector<std::string> combinations_include_first = CombinationIterator(characters_except_first, combinationLength - 1).getCombinations_();
            std::vector<std::string> combinations_except_first = CombinationIterator(characters_except_first, combinationLength).getCombinations_();
            for (size_t i = 0; i < combinations_include_first.size(); i++)
            {
                combinations_.emplace_back(characters.substr(0, 1).append(combinations_include_first.at(i)));
            }
            combinations_.insert(combinations_.end(), combinations_except_first.begin(), combinations_except_first.end());
        }
    }

    std::string next() {
        if (hasNext())
            return combinations_.at(offset_++);
        return std::string();
    }

    bool hasNext() {
        return offset_ < combinations_.size();
    }

    const std::vector<std::string> getCombinations_() { return combinations_; }
private:
    std::vector<std::string> combinations_;
    size_t offset_ = 0;
#endif
};

/**
 * Your CombinationIterator object will be instantiated and called as such:
 * CombinationIterator* obj = new CombinationIterator(characters, combinationLength);
 * string param_1 = obj->next();
 * bool param_2 = obj->hasNext();
 */

class StreamChecker {
public:
    StreamChecker(std::vector<std::string>& words) {

    }

    bool query(char letter) {

    }
};

/**
 * Your StreamChecker object will be instantiated and called as such:
 * StreamChecker* obj = new StreamChecker(words);
 * bool param_1 = obj->query(letter);
 */