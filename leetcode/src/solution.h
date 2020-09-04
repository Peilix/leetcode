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
#include <numeric>

#include "node.h"
#include "list_node.h"
#include "tree_node.h"

#define ROOT_TO_LEAF false

class Solution
{
public:
    static int twoCitySchedCost(std::vector<std::vector<int>>& costs);
    static std::vector<std::vector<int>> reconstructQueue(std::vector<std::vector<int>>& people);
    static int change(int amount, std::vector<int> coins);
    static bool isPowerOfTwo(int n);
    static bool isPowerOfFour(int num);
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
    static int rob(std::vector<int> nums);
    static int rob(TreeNode* root);
    static int diameterOfBinaryTree(TreeNode* root);
    static std::vector<int> findDuplicates(std::vector<int>& nums);
    static std::vector<int> findDisappearedNumbers(std::vector<int>& nums);
    static std::vector<std::vector<int>> verticalOrder(TreeNode* root);
    static std::vector<std::vector<int>> verticalTraversal(TreeNode* root);
    static std::vector<int> inorderTraversal(TreeNode* root);
    static void recoverTree(TreeNode* root);
    static bool hasPathSum(TreeNode* root, int sum); 
#if ROOT_TO_LEAF
    std::vector<std::vector<int>> pathSum(TreeNode* root, int sum);
#else
    static int pathSum(TreeNode* root, int sum);
#endif
    static int closestValue(TreeNode* root, double target);
    static int orangesRotting(std::vector<std::vector<int>>& grid);
    static int titleToNumber(std::string s);
    static int countBinarySubstrings(std::string s);
    static std::string makeGood(std::string s);
    static char findKthBit(int n, int k);
    static int maxNonOverlapping(std::vector<int>& nums, int target);
    static std::vector<std::vector<int>> generate(int numRows);
    static std::vector<int> getRow(int rowIndex);
    static Node* cloneGraph(Node* node);
    static ListNode* addTwoNumbers(ListNode* l1, ListNode* l2);
    static std::string addStrings(std::string num1, std::string num2);
    static std::string multiply(std::string num1, std::string num2);
    static bool isValid(std::string s);
    static int longestPalindrome(std::string s);
    static bool canPermutePalindrome(std::string s);
    static int removeBoxes(std::vector<int>& boxes);
    static int eraseOverlapIntervals(std::vector<std::vector<int>>& intervals);
    static std::vector<int> findPermutation(std::string s);
    static bool threeConsecutiveOdds(std::vector<int>& arr);
    static int minOperations(int n);
    static int maxDistance(std::vector<int>& position, int m);
    static int minDays(int n);
    static int maxProfit_1st(std::vector<int>& prices); // leetcode 121
    static int maxProfit_2nd(std::vector<int>& prices); // leetcode 122
    static int maxProfit_3rd(std::vector<int>& prices); // leetcode 123
    static int maxProfit_4th(std::vector<int>& prices); // leetcode 188
    static int maxProfit_5th(std::vector<int>& prices); // leetcode 309
    static std::vector<int> distributeCandies(int candies, int num_people);
    static std::vector<int> numsSameConsecDiff(int N, int K);
    static std::string toGoatLatin(std::string S);
    static void reorderList(ListNode* head);
    static std::vector<std::vector<char>> updateBoard(std::vector<std::vector<char>>& board, std::vector<int>& click);
    static int minDepth(TreeNode* root);
    static int maxDepth(TreeNode* root);
    static std::vector<int> sortArrayByParity(std::vector<int>& A);
    static bool isBalanced(TreeNode* root);
    static int numOfMinutes(int n, int headID, std::vector<int>& manager, std::vector<int>& informTime);
    static bool isCousins(TreeNode* root, int x, int y);
    static bool judgePoint24(std::vector<int>& nums);
    static int rangeBitwiseAnd(int m, int n);
    static bool repeatedSubstringPattern(std::string s);
    static int sumOfLeftLeaves(TreeNode* root);
    static std::vector<std::vector<int>> findSubsequences(std::vector<int>& nums);
    static int mincostTickets(std::vector<int>& days, std::vector<int>& costs);
    static std::vector<std::string> letterCombinations(std::string digits);
    static std::vector<std::string> fizzBuzz(int n);
    static ListNode* sortList(ListNode* head);
    static std::vector<int> countBits(int num);
    static TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2);
    static std::vector<int> findRightInterval(std::vector<std::vector<int>>& intervals);
    static std::string intToRoman(int num);
    static int romanToInt(std::string s);
    static bool judgeCircle(std::string moves);
    static int rand10();
    static bool hasCycle(ListNode* head);
    static ListNode* getIntersectionNode(ListNode* headA, ListNode* headB);
    static std::vector<std::string> findRestaurant(std::vector<std::string>& list1, std::vector<std::string>& list2);
    static std::string shortestParlindrome(std::string s);
    static std::vector<int> pancakeSort(std::vector<int>& A);
    static bool containsPattern(std::vector<int>& arr, int m, int k);
    static int getMaxLen(std::vector<int>& nums);
    static int largestComponentSize(std::vector<int>& A);
    static TreeNode* deleteNode(TreeNode* root, int key);
    static std::string largestTimeFromDigits(std::vector<int>& A);
    static std::vector<std::vector<int>> permute(std::vector<int>& nums);
    static bool containsNearbyAlmostDuplicate(std::vector<int>& nums, int k, int t);
    static std::vector<std::string> binaryTreePaths(TreeNode* root);
    static std::vector<int> partitionLabels(std::string S);
private:
};