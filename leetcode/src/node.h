#pragma once
#include <vector>

// Definition for a Node.
class Node
{
public:
    int val;
    Node* prev;
    Node* next;
    Node* child;
	std::vector<Node*> neighbors;

	Node() {
		val = 0;
		neighbors = std::vector<Node*>();
	}

	Node(int _val) {
		val = _val;
		neighbors = std::vector<Node*>();
	}

	Node(int _val, std::vector<Node*> _neighbors) {
		val = _val;
		neighbors = _neighbors;
	}
};
