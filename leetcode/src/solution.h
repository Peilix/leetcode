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
	int minimumMoves(std::vector<int> &arr);
	int uniquePathsIII(std::vector<std::vector<int> > &grid);
	int firstMissingPositive(std::vector<int> &nums);
	std::vector<std::vector<int> >
	allPathsSourceTarget(std::vector<std::vector<int> > &graph);
	std::vector<std::vector<std::string> >
	accountsMerge(std::vector<std::vector<std::string> > &accounts);
	bool canReach(std::vector<int> &arr, int start);
	int nthMagicalNumber(int n, int a, int b);
	std::vector<int>
	findOrder(int numCourses,
		  std::vector<std::vector<int> > &prerequisites);
	bool canReach(std::string s, int minJump, int maxJump);
	int maximumGood(std::vector<std::vector<int> > &statements);
	int findRadius(std::vector<int> &houses, std::vector<int> &heaters);
	int largestRectangleArea(std::vector<int> &heights);
	int widthOfBinaryTree(TreeNode *root);
	std::vector<std::string> summaryRanges(std::vector<int> &nums);
	int getKth(int lo, int hi, int k);
	std::string minRemoveToMakeValid(std::string s);
	int scoreOfParentheses(std::string s);
	int minDominoRotations(std::vector<int> &tops,
			       std::vector<int> &bottoms);
	std::vector<int> busiestServers(int k, std::vector<int> &arrival,
					std::vector<int> &load);
	bool hasAllCodes(std::string s, int k);
	std::vector<int>
	countRectangles(std::vector<std::vector<int> > &rectangles,
			std::vector<std::vector<int> > &points);
	int consecutiveNumbersSum(int n);
<<<<<<< HEAD
	std::vector<int>
	fullBloomFlowers(std::vector<std::vector<int> > &flowers,
			 std::vector<int> &persons);
	int maxLength(std::vector<int> &nums);
	int maximumSum(std::vector<int> &nums);
	int numTilePossibilities(std::string tiles);
	long long countOfSubstrings(std::string word, int k);
	int numberOfSubstrings(std::string s);
=======
	std::vector<int> fullBloomFlowers(std::vector<std::vector<int> > &flowers,
				     std::vector<int> &persons);
>>>>>>> 814e0a7adb23eebc5e1c35397dc0cc60509111e7
};
