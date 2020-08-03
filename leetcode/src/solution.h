#pragma once
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <stack>
#include <functional>
#include <algorithm>
#include <sstream>

#include "list_node.h"
#include "tree_node.h"
class Solution
{
public:
    static int twoCitySchedCost(std::vector<std::vector<int>>& costs);
    static std::vector<std::vector<int>> reconstructQueue(std::vector<std::vector<int>>& people);
    static int change(int amount, std::vector<int> coins);
    static bool isPowerOfTwo(int n);
    static bool isSubsequence(std::string s, std::string t);
    static void sortColors(std::vector<int>& nums);
    static std::vector<int> largestDivisibleSubset(std::vector<int>& nums);
    static int findCheapestPrice(int n, std::vector<std::vector<int>>& flights, int src, int dst, int K);
    static std::string validIPAddress(std::string IP);
    static void solve(std::vector<std::vector<char>>& board);
    static int hIndex(std::vector<int>& citations);
    static std::string longestDupSubstring(std::string S);
    static std::string getPermutation(int n, int k);
    static int calculateMinimumHP(std::vector<std::vector<int>>& dungeon);
    static int singleNumber(std::vector<int>& nums);
    static std::string longestDiverseString(int a, int b, int c);
    static std::vector<std::vector<int>> merge(std::vector<std::vector<int>>& intervals);
    static int findDuplicate(std::vector<int>& nums);
    static std::vector<std::string> findItinerary(std::vector<std::vector<std::string>>& tickets);
    static int uniquePaths(int m, int n);
    static int arrangeCoins(int n);
    static std::vector<int> prisonAfterNDays(std::vector<int>& cells, int N);
    static int hammingDistance(int x, int y);
    static std::vector<int> plusOne(std::vector<int>& digits);
    static bool isUgly(int num);
    static int nthUglyNumber(int n);
    static int islandPerimeter(std::vector<std::vector<int>>& grid);
    static std::vector<std::vector<int>> threeSum(std::vector<int>& nums);
    static std::vector<std::vector<int>> subsets(std::vector<int>& nums);
    static uint32_t reverseBits(uint32_t n);
    static double angleClock(int hour, int minutes);
    static std::string reverseWords(std::string s);
    static char* reverseWords(char* s);
    static double myPow(double x, int n);
    static int superPow(int a, std::vector<int>& b);
    static std::vector<int> topKFrequent(std::vector<int>& nums, int k);
    static std::string addBinary(std::string a, std::string b);
    static bool canFinish(int numCourses, std::vector<std::vector<int>>& prerequisites);
    static std::vector<int> findOrder(int numCourses, std::vector<std::vector<int>>& prerequisites);
    static ListNode* reverseBetween(ListNode* head, int m, int n);
    static bool exist(std::vector<std::vector<char>>& board, std::string word);
    static std::string convert(std::string s, int numRows);
    static char* convert(char* s, int numRows);
    static int numWaterBottles(int numBottles, int numExchange);
    static std::vector<int> singleNumbers(std::vector<int>& nums);
    static int* singleNumber(int* nums, int numsSize, int* returnSize);
    static std::vector<std::vector<int>> allPathsSourceTarget(std::vector<std::vector<int>>& graph);
    static int findMin(std::vector<int>& nums);
    static TreeNode* buildTree(std::vector<int>& inorder, std::vector<int>& postorder);
    static int leastInterval(std::vector<char>& tasks, int n);
    static std::string reorganizeString(std::string S);
    static void merge(std::vector<int>& nums1, int m, std::vector<int>& nums2, int n);
    static std::vector<int> sortedSquares(std::vector<int>& A);
    static ListNode* mergeTwoLists(ListNode* l1, ListNode* l2);
    static ListNode* mergeKLists(std::vector<ListNode*>& lists);
    static std::vector<std::string> wordBreak(std::string s, std::vector<std::string>& wordDict);
    static int integerBreak(int n);
    static int climbStairs(int n);
    static std::string getHint(std::string secret, std::string guess);
    static bool detectCapitalUse(std::string word);
    static std::vector<int> smallestRange(std::vector<std::vector<int>>& nums);
    static int countGoodTriplets(std::vector<int>& arr, int a, int b, int c);
    static int getWinner(std::vector<int>& arr, int k);
    static int minSwaps(std::vector<std::vector<int>>& grid);
    static int maxSum(std::vector<int>& nums1, std::vector<int>& nums2);
    static bool isPalindrome(std::string s);
private:

};