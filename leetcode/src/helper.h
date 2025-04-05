#pragma once

#include <vector>
#include <optional>

#include "tree_node.h"

class TreeBuilder {
    public:
	TreeBuilder() = default;
	~TreeBuilder() = default;
	TreeNode *buildFromLevelOrder(std::vector<std::optional<int> > data)
	{
		if (data.empty()) {
			return nullptr;
		}
		std::queue<TreeNode *> nodes_queue;
		for (const auto &val : data) {
			if (val.has_value()) {
				nodes_queue.push(new TreeNode(val.value()));
			} else {
				nodes_queue.push(nullptr);
			}
		}
		TreeNode *root = nodes_queue.front();
		auto children_queue = nodes_queue;
		children_queue.pop();
		while (!children_queue.empty() && !nodes_queue.empty()) {
			if (nodes_queue.front() == nullptr) {
				nodes_queue.pop();
				continue;
			}
			nodes_queue.front()->left = children_queue.front();
			children_queue.pop();
			if (!children_queue.empty()) {
				nodes_queue.front()->right = children_queue.front();
				children_queue.pop();
			}
			nodes_queue.pop();
		}

		return root;
	}
	void destoryTree(TreeNode *root)
	{
		if (root == nullptr) {
			return;
		}
		destoryTree(root->left);
		destoryTree(root->right);
		delete root;
	}

    private:
};
