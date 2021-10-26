#pragma once
#include <vector>
#include <string>

#include "tree_node.h"

class Solution {
    public:
	int fib(int n);
	int tribonacci(int n);
	inline int climbStairs(int n)
	{
		return fib(n + 1);
	}
	int minCostClimbingStairs(std::vector<int> &cost);
	int maxLength(std::vector<std::string> &arr);
	int rob(std::vector<int> &nums);
	int deleteAndEarn(std::vector<int> &nums);
	std::string breakPalindrome(std::string palindrome);
	bool canJump(std::vector<int> &nums);
	int jump(std::vector<int> &nums);
	int shortestPath(std::vector<std::vector<int> > &grid, int k);
	int maxSubArray(std::vector<int> &nums);
	int maxSubarraySumCircular(std::vector<int> &nums);
	int maxProduct(std::vector<int> &nums);
	int getMaxLen(std::vector<int> &nums);
	int numUniqueEmails(std::vector<std::string> &emails);
	std::vector<int> sortArrayByParityII(std::vector<int> &nums);
	int numSquares(int n);
	std::vector<std::vector<int> >
	updateMatrix(std::vector<std::vector<int> > &mat);
	bool canPartitionKSubsets(std::vector<int> &nums, int k);
	int longestCommonSubsequence(std::string text1, std::string text2);
	bool exist(std::vector<std::vector<char> > &board, std::string word);
	std::vector<std::string>
	findWords(std::vector<std::vector<char> > &board,
		  std::vector<std::string> &words);
	std::vector<double> medianSlidingWindow(std::vector<int> &nums, int k);
	int minimumDifference(std::vector<int> &nums);
	std::string longestDupSubstring(std::string s);
	double findMedianSortedArrays(std::vector<int> &nums1,
				      std::vector<int> &nums2);
	std::string frequencySort(std::string s);
	int findMin(std::vector<int> &nums);
	std::vector<int> constructRectangle(int area);
	int countNodes(TreeNode *root);
};
