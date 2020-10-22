#pragma once
#include <string>
#include <unordered_map>
#include <map>

#include "tree_node.h"
#include <stack>
#include <queue>

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
    struct TrieNode
    {
        std::map<char, TrieNode*> children;
        bool is_end = false;
    };
    Trie() {
        root = new TrieNode();
    }
    ~Trie() {
        if (root->children.empty())
            delete root;
    }

    /** Inserts a word into the trie. */
    void insert(std::string word) {
        TrieNode* curr = root;
        for (auto letter : word)
        {
            if (curr->children.find(letter) == curr->children.end()) {
                curr->children[letter] = new TrieNode();
            }
            curr = curr->children[letter];
        }
        curr->is_end = true;
    }

    /** Returns if the word is in the trie. */
    bool search(std::string word) {
        TrieNode* curr = root;
        for (auto letter : word)
        {
            if (curr->children.find(letter) == curr->children.end()) {
                return false;
            }
            curr = curr->children[letter];
        }
        return curr->is_end;
    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(std::string prefix) {
        TrieNode* curr = root;
        for (auto letter : prefix)
        {
            if (curr->children.find(letter) == curr->children.end()) {
                return false;
            }
            curr = curr->children[letter];
        }
        return true;
    }

private:
    TrieNode* root;
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */

class WordDictionary {
    struct TrieNode
    {
        std::map<char, TrieNode*> children;
        bool is_end = false;
    };
    TrieNode* root;
public:
    /** Initialize your data structure here. */
    WordDictionary() {
        root = new TrieNode();
    }

    /** Adds a word into the data structure. */
    void addWord(std::string word) {
        TrieNode* curr = root;
        for (auto letter : word)
        {
            if (curr->children.find(letter) == curr->children.end()) {
                curr->children[letter] = new TrieNode();
            }
            curr = curr->children[letter];
        }
        curr->is_end = true;
    }

    /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
    bool search(std::string word) {
        return searchRecursive(root, word, 0);
    }
private:
    bool searchRecursive(TrieNode* curr, std::string& word, size_t index) {
        if (index == word.size())
            return curr->is_end;
        char letter = word[index];
        if (letter == '.')
        {
            for (const auto& e: curr->children)
                if (searchRecursive(e.second, word, index + 1))
                    return true;
        }
        else
        {
            if (curr->children.find(letter) == curr->children.end())
                return false;
            return searchRecursive(curr->children[letter], word, index + 1);
        }
        return false;
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


class BSTIterator {
    std::stack<TreeNode*> stack_;
public:
    BSTIterator(TreeNode* root) {
        while (root) {
            stack_.push(root);
            root = root->left;
        }
    }

    /** @return the next smallest number */
    int next() {
        TreeNode* curr = stack_.top();
        stack_.pop();
        int result = curr->val;
        curr = curr->right;
        while (curr) {
            stack_.push(curr);
            curr = curr->left;
        }
        return result;
    }

    /** @return whether we have a next smallest number */
    bool hasNext() {
        return stack_.empty();
    }
};

/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator* obj = new BSTIterator(root);
 * int param_1 = obj->next();
 * bool param_2 = obj->hasNext();
 */

class CBTInserter {
    TreeNode* root_ = nullptr;
    std::queue<TreeNode*> queue_;
public:
    CBTInserter(TreeNode* root) : root_(root) {
        queue_.push(root);
        while (!queue_.empty())
        {
            root = queue_.front();
            if (root->left && root->right) {
                queue_.pop();
                queue_.push(root->left);
                queue_.push(root->right);
            }
            else
                break;
        }
    }

    int insert(int v) {
        TreeNode* new_node = new TreeNode(v);
        TreeNode* curr = queue_.front();
        if (curr->left == nullptr) {
            curr->left = new_node;
        }
        else {
            curr->right = new_node;
            queue_.pop();
        }
        return curr->val;
    }

    TreeNode* get_root() {
        return root_;
    }
};

/**
 * Your CBTInserter object will be instantiated and called as such:
 * CBTInserter* obj = new CBTInserter(root);
 * int param_1 = obj->insert(v);
 * TreeNode* param_2 = obj->get_root();
 */

class ThroneInheritance {
    struct person
    {
        std::string name;
        bool dead = false;
        std::vector<person*> children;
        person(std::string _name) : name(_name) {}
    };
    person* king;
    std::map<std::string, person*> map;
    std::vector<std::string> inheritanceOrder_;

    void preorderTraversal(person* current) {
        if (!current->dead)
            inheritanceOrder_.push_back(current->name);
        for (const auto& child : current->children)
        {
            preorderTraversal(child);
        }
    }
public:
    ThroneInheritance(std::string kingName) {
        king = new person(kingName);
        map[kingName] = king;
    }

    void birth(std::string parentName, std::string childName) {
        auto newborn = new person(childName);
        map[childName] = newborn;
        map[parentName]->children.push_back(newborn);
    }

    void death(std::string name) {
        map[name]->dead = true;
    }

    std::vector<std::string> getInheritanceOrder() {
        inheritanceOrder_.clear();
        preorderTraversal(king);
        return inheritanceOrder_;
    }
};

/**
 * Your ThroneInheritance object will be instantiated and called as such:
 * ThroneInheritance* obj = new ThroneInheritance(kingName);
 * obj->birth(parentName,childName);
 * obj->death(name);
 * vector<string> param_3 = obj->getInheritanceOrder();
 */

class RecentCounter {
    std::queue<int> q_;
public:
    RecentCounter() {

    }

    int ping(int t) {
        q_.push(t);
        while (q_.front() < t - 3000)
        {
            q_.pop();
        }
        return q_.size();
    }
};

/**
 * Your RecentCounter object will be instantiated and called as such:
 * RecentCounter* obj = new RecentCounter();
 * int param_1 = obj->ping(t);
 */
 /**
  * Definition for a binary tree node.
  * struct TreeNode {
  *     int val;
  *     TreeNode *left;
  *     TreeNode *right;
  *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
  * };
  */
class Codec {
public:

    // Encodes a tree to a single string.
    std::string serialize(TreeNode* root) {
        std::vector<std::string> strvec;
        //level order root
        {
            std::queue<TreeNode*> q;
            q.push(root);
            while (!q.empty())
            {
                TreeNode* curr = q.front();
                q.pop();
                if (curr == nullptr)
                    strvec.push_back("null");
                else {
                    strvec.push_back(std::to_string(curr->val));
                    q.push(curr->left);
                    q.push(curr->right);
                }
            }
        }
        std::string result = "[";

        for (auto str : strvec)
        {
            result.append(str);
            result.push_back(',');
        }
        result.back() = ']';
        return result;
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(std::string data) {
        if (data.size() <= 2)
            return nullptr;
        auto split = [](const std::string& str, const char& chr) -> std::vector<std::string> {
            std::vector<std::string> res;
            auto last = 0;
            for (size_t i = 0; i < str.length(); i++)
            {
                if (str[i] == chr)
                {
                    if (last != i)
                        res.emplace_back(std::string(str.begin() + last, str.begin() + i));
                    last = i + 1;
                }
            }
            if (last != str.length())
                res.emplace_back(std::string(str.begin() + last, str.end()));
            return res;
        };
        std::vector<std::string> strvec = split(data, ',');
        strvec[0].erase(strvec[0].begin());
        strvec.back().resize(strvec.back().size() - 1);
        TreeNode dummy;
        std::queue<TreeNode*> q;
        q.push(&dummy);
        bool right = true;
        for (auto str : strvec)
        {
            while (q.front() == nullptr)
                q.pop();
            auto parent = q.front();
            TreeNode* curr = nullptr;
            if (str != "null")
                curr = new TreeNode(std::stoi(str));
            if (right) {
                parent->right = curr;
                q.pop();
            }
            else
            {
                parent->left = curr;
            }
            q.push(curr);
            right = !right;
        }
        return dummy.right;
    }
};

// Your Codec object will be instantiated and called as such:
// Codec* ser = new Codec();
// Codec* deser = new Codec();
// string tree = ser->serialize(root);
// TreeNode* ans = deser->deserialize(tree);
// return ans;