#include "solution.h"

#include <cassert>
#include <algorithm>
#include <numeric>
#include <set>
#include <map>
#include <unordered_set>
#include <ranges>
#include <queue>
#include <stack>
#include <unordered_map>
#include <functional>

int Solution::fib(int n)
{
	if (n < 2)
		return n;
	int ret = 0;
	for (int i = 2, a = 0, b = 1; i <= n; i++) {
		ret = a + b;
		a = b;
		b = ret;
	}
	return ret;
}

int Solution::tribonacci(int n)
{
	if (n < 3)
		return static_cast<int>(n != 0);
	int ret = 0;
	for (int i = 3, a = 0, b = 1, c = 1; i <= n; i++) {
		ret = a + b + c;
		a = b;
		b = c;
		c = ret;
	}
	return ret;
}

int Solution::minCostClimbingStairs(std::vector<int> &cost)
{
	std::vector<int> dp(cost.size() + 1, 0);
	for (int i = 2; i < dp.size(); i++)
		dp[i] = std::min(dp[i - 1] + cost[i - 1],
				 dp[i - 2] + cost[i - 2]);
	return dp.back();
}

int Solution::maxLength(std::vector<std::string> &arr)
{
	std::vector<int> masks(arr.size(), 0);
	for (size_t i = 0; i < arr.size(); i++) {
		auto &mask = masks[i];
		for (auto chr : arr[i]) {
			if ((mask & (1 << (chr - 'a'))) != 0) {
				mask = 0;
				break;
			}
			mask |= (1 << (chr - 'a'));
		}
	}

	int bitmask = 0;
	int bitmask_upper_bound = 1 << arr.size();
	size_t ret = 0;
	for (; bitmask < bitmask_upper_bound; bitmask++) {
		size_t total_length = 0;
		int total_mask = 0;
		for (int i = 0; i < arr.size(); i++) {
			if ((bitmask & (1 << i)) != 0) {
				total_length += arr[i].length();
				if (masks[i] == 0 || 26 < total_length) {
					break;
				}
				if (total_mask + masks[i] !=
				    (total_mask | masks[i])) {
					break;
				}
				total_mask |= masks[i];
				ret = std::max(ret, total_length);
			}
		}
	}
	return static_cast<int>(ret);
}

int Solution::rob(std::vector<int> &nums)
{
	if (nums.empty())
		return 0;
	if (1 == nums.size())
		return nums[0];
	std::vector<int> dp(nums.size(), 0);
	dp[0] = nums[0];
	dp[1] = std::max(nums[0], nums[1]);
	for (size_t i = 2; i < nums.size(); i++) {
		dp[i] = std::max(dp[i - 2] + nums[i], dp[i - 1]);
	}
	return dp.back();
}

int Solution::deleteAndEarn(std::vector<int> &nums)
{
	std::map<int, int> counts;
	for (auto num : nums)
		counts[num]++;
	int min_num = counts.cbegin()->first;
	int max_num = counts.crbegin()->first;

	if (max_num < min_num + 2)
		return std::max(min_num * counts[min_num],
				max_num * counts[max_num]);

	std::vector<int> dp(max_num - min_num + 1, 0);
	dp[0] = counts[min_num] * min_num;
	dp[1] = std::max(counts[min_num], counts[min_num + 1] * (min_num + 1));
	dp[2] = std::max(dp[0] + counts[min_num + 2] * (min_num + 2), dp[1]);
	for (int i = min_num + 3; i <= max_num; i++) {
		int index = i - min_num;
		dp[index] = std::max(
			std::max(dp[index - 2] + counts[i] * i,
				 dp[index - 3] + counts[i - 1] * (i - 1)),
			dp[index - 1]);
	}
	return dp.back();
}

std::string Solution::breakPalindrome(std::string palindrome)
{
	if (palindrome.length() < 2) {
		return std::string();
	}
	for (size_t i = 0; i < palindrome.length() / 2; i++) {
		if (palindrome[i] != 'a') {
			palindrome[i] = 'a';
			return palindrome;
		}
	}
	palindrome.back() = 'b';
	return palindrome;
}

bool Solution::canJump(std::vector<int> &nums)
{
	int max_reach = 0;
	for (int i = 0; i < nums.size() - 1; i++) {
		max_reach = std::max(max_reach, nums[i] + i);
		if (max_reach <= i)
			return false;
	}
	return true;
}

int Solution::jump(std::vector<int> &nums)
{
	std::vector<int> dp(nums.size(), INT_MAX);
	dp[0] = 0;
	for (size_t i = 0; i < nums.size(); i++)
		for (size_t j = 1; j <= nums[i] && i + j < nums.size(); j++)
			dp[i + j] = std::min(dp[i] + 1, dp[i + j]);
	return dp.back();
}
// TODO:  BFS or DP (BFS exercise)
int Solution::shortestPath(std::vector<std::vector<int> > &grid, int k)
{
	constexpr int error_code = -1;
	if (grid.empty() || grid.back().empty())
		return error_code;
	const int m = static_cast<int>(grid.size());
	const int n = static_cast<int>(grid.back().size());
	if (m + n - 1 < k)
		return m + n - 2;
	int ret = m * n + 1;

	struct Step {
		int x;
		int y;
		int r;
		int s;
	};
	std::queue<Step> q;
	q.push({ 0, 0, k, 0 });
	std::vector<std::vector<std::vector<bool> > > visited(
		grid.size(),
		std::vector<std::vector<bool> >(
			grid.back().size(), std::vector<bool>(k + 1, false)));
	std::vector<std::pair<int, int> > moves{
		{ -1, 0 }, { 0, -1 }, { 0, 1 }, { 1, 0 }
	};
	visited[0][0][grid[0][0]] = true;
	while (!q.empty()) {
		auto pace = q.front();
		q.pop();
		if (pace.x == m - 1 && pace.y == n - 1) {
			return pace.s;
		}
		for (const auto &move : moves) {
			Step step{ pace.x + move.first, pace.y + move.second,
				   pace.r - grid[pace.x][pace.y], pace.s + 1 };
			if (0 <= step.x && step.x < m && 0 <= step.y &&
			    step.y < n && 0 <= step.r &&
			    !visited[step.x][step.y][step.r]) {
				q.push(step);
				visited[step.x][step.y][step.r] = true;
			}
		}
	}
	return error_code;
}

/*
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
A subarray is a contiguous part of an array.
*/
int Solution::maxSubArray(std::vector<int> &nums)
{
	assert(!nums.empty());
	if (nums.size() < 2)
		return nums.back();
	auto max_sum = nums[0];
	auto ret = nums[0];
	for (auto it = nums.begin() + 1; it != nums.end(); it++) {
		max_sum = max_sum < 0 ? *it : max_sum + *it;
		ret = ret < max_sum ? max_sum : ret;
	}
	return ret;
}

/*
Given a circular integer array nums of length n, return the maximum possible sum of a non-empty subarray of nums.
A circular array means the end of the array connects to the beginning of the array. Formally, the next element of nums[i] is nums[(i + 1) % n] and the previous element of nums[i] is nums[(i - 1 + n) % n].
A subarray may only include each element of the fixed buffer nums at most once. Formally, for a subarray nums[i], nums[i + 1], ..., nums[j], there does not exist i <= k1, k2 <= j with k1 % n == k2 % n.
*/
int Solution::maxSubarraySumCircular(std::vector<int> &nums)
{
	int sum = std::accumulate(nums.begin(), nums.end(), 0);
	assert(!nums.empty());
	if (nums.size() < 2)
		return nums.back();
	auto max_sum = nums[0];
	auto min_sum = nums[0];
	auto ret = nums[0];
	for (auto it = nums.begin() + 1; it != nums.end(); it++) {
		min_sum = 0 < min_sum ? *it : min_sum + *it;
		ret = min_sum < ret ? min_sum : ret;
	}
	ret = sum - ret;

	if (ret == 0) // empty subarray
		ret = nums[0];
	for (auto it = nums.begin() + 1; it != nums.end(); it++) {
		max_sum = max_sum < 0 ? *it : max_sum + *it;
		ret = ret < max_sum ? max_sum : ret;
	}

	return ret;
}

int Solution::maxProduct(std::vector<int> &nums)
{
	int ret = 0;
	//auto const ints = { 0, 1, 2, 3, 4, 5 };
	//auto even = [](int i) { return 0 == i % 2; };
	//auto square = [](int i) { return i * i; };
	//for (int i : ints | std::ranges::views::filter(even) |
	//		     std::ranges::views::transform(square)) {
	//	//std::cout << i << ' ';
	//	ret = i;
	//}
	return ret;
}

int Solution::getMaxLen(std::vector<int> &nums)
{
	int ret = 0;
	return ret;
}

int Solution::numUniqueEmails(std::vector<std::string> &emails)
{
	std::unordered_set<std::string> set_of_emails;
	for (const auto &email : emails) {
		std::string local_name;
		size_t j = 0;
		for (size_t i = 0; i < email.length(); i++) {
			if (email[i] == '@') {
				j = i;
				break;
			} else if (email[i] == '+') {
				break;
			} else if (email[i] == '.') {
				continue;
			} else {
				local_name.push_back(email[i]);
			}
		}
		while (email[j] != '@') {
			j++;
		}
		set_of_emails.insert(local_name + email.substr(j));
	}
	return static_cast<int>(set_of_emails.size());
}

std::vector<int> Solution::sortArrayByParityII(std::vector<int> &nums)
{
	for (int i = 0, j = 1; i < nums.size() - 1 && j < nums.size();
	     i += 2, j += 2) {
		while (i < nums.size() - 1 && ((i ^ nums[i]) & 1) == 0)
			i += 2;
		while (j < nums.size() && ((j ^ nums[j]) & 1) == 0)
			j += 2;
		if (i < nums.size() - 1 && j < nums.size())
			std::swap(nums[i], nums[j]);
	}
	return nums;
}

int Solution::numSquares(int n)
{
	assert(0 < n);
	std::vector<int> f(n + 1);
	for (int i = 1; i <= n; i++) {
		int minn = INT_MAX;
		for (int j = 1; j * j <= i; j++)
			minn = std::min(minn, f[i - j * j]);
		f[i] = minn + 1;
	}
	return f[n];
}

std::vector<std::vector<int> >
Solution::updateMatrix(std::vector<std::vector<int> > &mat)
{
	return std::vector<std::vector<int> >();
}

bool Solution::canPartitionKSubsets(std::vector<int> &nums, int k)
{
	int sum = std::accumulate(nums.begin(), nums.end(), 0);
	if (sum % k != 0)
		return false;
	int n = sum / k;
	std::sort(nums.begin(), nums.end());
	if (n < nums.back())
		return false;
	std::vector<bool> mark(nums.size(), false);

	for (size_t i = nums.size(); 0 < i--;) {
		int left = 0;
		int right = static_cast<int>(i) - 1;
		while (left < right) {
			int mid = left + ((right - left) >> 1);
			if (nums[mid])
				//TOOD
				mid++;
		}
	}

	return false;
}

int Solution::longestCommonSubsequence(std::string text1, std::string text2)
{
	return 0;
}

bool Solution::exist(std::vector<std::vector<char> > &board, std::string word)
{
	std::vector<std::vector<bool> > visited(
		board.size(), std::vector<bool>(board[0].size(), false));
	std::vector<std::vector<int> > adjacents = {
		{ -1, 0 }, { 0, -1 }, { 0, 1 }, { 1, 0 }
	};
	auto backtrace = [&](auto &&f, int row, int column, int index) -> bool {
		if (row < 0 || board.size() <= row || column < 0 ||
		    board[0].size() <= column ||
		    board[row][column] != word[index] ||
		    visited[row][column] == true)
			return false;
		if (index == word.size() - 1) {
			return true;
		}
		visited[row][column] = true;
		for (const auto &adjacent : adjacents) {
			int i = row + adjacent[0];
			int j = column + adjacent[1];
			if (f(f, i, j, index + 1)) {
				visited[row][column] = false;
				return true;
			}
		}
		visited[row][column] = false;
		return false;
	};
	for (int i = 0; i < board.size(); i++) {
		for (int j = 0; j < board[0].size(); j++) {
			if (backtrace(backtrace, i, j, 0))
				return true;
		}
	}
	return false;
}

std::vector<std::string>
Solution::findWords(std::vector<std::vector<char> > &board,
		    std::vector<std::string> &words)
{
	std::vector<std::string> ret;

	std::vector<std::vector<bool> > visited(
		board.size(), std::vector<bool>(board[0].size(), false));
	std::vector<std::vector<int> > adjacents = {
		{ -1, 0 }, { 0, -1 }, { 0, 1 }, { 1, 0 }
	};
	class Trie {
	    public:
		void insert(std::string){};
	};
	Trie trie;
	for (auto &&word : words) {
		// Trie insert(word);
		trie.insert(word);
	}
	auto backtrace = [&](auto &&f, int row, int column, int index) -> void {
		if (row < 0 || board.size() <= row || column < 0 ||
		    board[0].size() <= column || visited[row][column] == true)
			return;
		if (true) // not in trie
			return;
		// if in trie and is_end
		//ret.emplace_back();
		visited[row][column] = true;
		for (const auto &adjacent : adjacents) {
			int i = row + adjacent[0];
			int j = column + adjacent[1];
			f(f, i, j, index + 1);
		}
		visited[row][column] = false;
	};
	return ret;
}

std::vector<double> Solution::medianSlidingWindow(std::vector<int> &nums, int k)
{
	class DualHeap {
	    public:
		DualHeap(int k) : k_(k) {};
		void insert(int num)
		{
			if (max_heap_.empty() || num <= max_heap_.top()) {
				max_heap_.push(num);
			} else {
				min_heap_.push(num);
			}
			rebalance();
		}
		void erase(int num)
		{
			if (num == max_heap_.top()) {
				max_heap_.pop();
				while (!max_heap_.empty() &&
				       0 < delayed_[max_heap_.top()]) {
					delayed_[max_heap_.top()]--;
					delayed_l_--;
					max_heap_.pop();
				};
			} else if (num == min_heap_.top()) {
				min_heap_.pop();
				while (!min_heap_.empty() &&
				       0 < delayed_[min_heap_.top()]) {
					delayed_[min_heap_.top()]--;
					delayed_r_--;
					min_heap_.pop();
				};
			} else if (num < max_heap_.top()) {
				delayed_[num]++;
				delayed_l_++;
			} else {
				delayed_[num]++;
				delayed_r_++;
			}
			rebalance();
		}
		double median()
		{
			return (k_ & 1) == 1 ?
				       static_cast<double>(max_heap_.top()) :
				       max_heap_.top() / 2.0 +
					       min_heap_.top() / 2.0;
		}

	    private:
		std::priority_queue<int> max_heap_;
		std::priority_queue<int, std::vector<int>, std::greater<int> >
			min_heap_;
		std::unordered_map<int, int> delayed_;
		int delayed_l_ = 0;
		int delayed_r_ = 0;
		int k_;
		void rebalance()
		{
			auto left_size = max_heap_.size() - delayed_l_;
			auto right_size = min_heap_.size() - delayed_r_;
			while (left_size < right_size ||
			       right_size + 1 < left_size) {
				if (left_size < right_size) {
					while (0 < delayed_[min_heap_.top()]) {
						delayed_[min_heap_.top()]--;
						delayed_r_--;
						min_heap_.pop();
					};
					max_heap_.push(min_heap_.top());
					min_heap_.pop();
					while (0 < delayed_[min_heap_.top()]) {
						delayed_[min_heap_.top()]--;
						delayed_r_--;
						min_heap_.pop();
					};
				} else {
					while (0 < delayed_[max_heap_.top()]) {
						delayed_[max_heap_.top()]--;
						delayed_l_--;
						max_heap_.pop();
					};
					min_heap_.push(max_heap_.top());
					max_heap_.pop();
					while (0 < delayed_[max_heap_.top()]) {
						delayed_[max_heap_.top()]--;
						delayed_l_--;
						max_heap_.pop();
					};
				}
				left_size = max_heap_.size() - delayed_l_;
				right_size = min_heap_.size() - delayed_r_;
			}
		}
	};

	std::vector<double> ret;
	DualHeap dh(k);
	for (size_t i = 0; i < k; i++) {
		dh.insert(nums[i]);
	}
	ret.push_back(dh.median());
	for (size_t i = 0; i + k < nums.size(); i++) {
		dh.insert(nums[i + k]);
		dh.erase(nums[i]);
		ret.push_back(dh.median());
	}
	return ret;
}

int Solution::minimumDifference(std::vector<int> &nums)
{
	return 0;
}

std::string Solution::longestDupSubstring(std::string s)
{
	std::string ret;
	using namespace std;
	constexpr int mod = 1000000007;
	constexpr int base = 26;

	auto doubleCheck = [](unordered_map<long, pair<int, int> > &mp,
			      long curVal, string cur, string &s) -> bool {
		auto [i, j] = mp[curVal];
		return s.substr(i, j - i + 1) == cur;
	};

	auto has_dup_substring = [&](std::string &s, int len) -> bool {
		unordered_map<long, pair<int, int> > mp;
		int l = 0, r = 0, n = static_cast<int>(s.size());
		long curVal = 0, mul = 1;
		for (int i = 1; i <= len; ++i) {
			mul = mul * base % mod;

			curVal = (curVal * base + s[r++]) % mod;
		}
		mp[curVal] = make_pair(l, r - 1);
		while (r < n) {
			curVal = (curVal * base - s[l++] * mul) % mod + mod;
			curVal = (curVal + s[r]) % mod;
			if (mp.find(curVal) != mp.end() &&
			    doubleCheck(mp, curVal, s.substr(l, len), s)) {
				ret = s.substr(l, len);
				return true;
			}
			mp[curVal] = make_pair(l, r);
			++r;
		}
		return false;
	};

	int left = 0;
	int right = static_cast<int>(s.length());
	while (left < right) {
		int mid = left + (right - left) / 2;
		if (has_dup_substring(s, mid))
			left = mid + 1;
		else
			right = mid;
	}
	return ret;
}

double Solution::findMedianSortedArrays(std::vector<int> &nums1,
					std::vector<int> &nums2)
{
	if (nums2.size() < nums1.size()) {
		std::swap(nums1, nums2);
	}
	auto m = nums1.size();
	auto n = nums2.size();
	auto total_len = m + n;
	assert(0 < m + n);
	if (m == 0)
		return n % 2 == 0 ? (nums2[n / 2 - 1] + nums2[n / 2]) / 2.0 :
				    nums2[n / 2];

	std::pair<int, int> medians;
	size_t left = 0;
	size_t right = m;

	size_t nums1_pivot = 0, nums2_pivot = 0;

	while (left < right) {
		nums1_pivot = left + (right - left) / 2;
		nums2_pivot = (total_len + 1) / 2 - nums1_pivot;

		int nums1_before_i =
			(nums1_pivot == 0 ? INT_MIN : nums1[nums1_pivot - 1]);
		int nums2_j = (nums2_pivot == n ? INT_MAX : nums2[nums2_pivot]);
		if (nums1_before_i < nums2_j)
			left = nums1_pivot + 1;
		else
			right = nums1_pivot;
	}

	nums1_pivot = left;
	nums2_pivot = (total_len + 1) / 2 - nums1_pivot;

	int nums1_before_i =
		(nums1_pivot == 0 ? INT_MIN : nums1[nums1_pivot - 1]);
	int nums1_i = (nums1_pivot == m ? INT_MAX : nums1[nums1_pivot]);
	int nums2_before_j =
		(nums2_pivot == 0 ? INT_MIN : nums2[nums2_pivot - 1]);
	int nums2_j = (nums2_pivot == n ? INT_MAX : nums2[nums2_pivot]);
	medians.first = std::max(nums1_before_i, nums2_before_j);
	medians.second = std::min(nums1_i, nums2_j);
	return (m + n) % 2 == 0 ? (medians.first + medians.second) / 2.0 :
				  medians.first;
}

std::string Solution::frequencySort(std::string s)
{
	std::unordered_map<char, int> count;
	std::for_each(s.cbegin(), s.cend(), [&](char c) { count[c]++; });
	std::map<int, std::vector<char> > occurrence_map;
	std::for_each(count.cbegin(), count.cend(),
		      [&](std::pair<int, int> it) {
			      occurrence_map[it.second].push_back(it.first);
		      });
	std::string ret;
	std::for_each(
		occurrence_map.crbegin(), occurrence_map.crend(),
		[&](std::pair<int, std::vector<char> > it) {
			for (auto c : it.second)
				ret.append(std::move(std::string(it.first, c)));
		});
	return ret;
}

int Solution::findMin(std::vector<int> &nums)
{
	assert(!nums.empty());
	auto recursive_find_min = [](auto &&self, std::vector<int> &nums,
				     size_t left, size_t right) -> int {
		if (left == right || nums[left] < nums[right])
			return nums[left];
		auto last = right;
		while (left < right) {
			auto mid = left + (right - left) / 2;
			if (nums[mid] < nums[last]) {
				right = mid;
			} else if (nums[mid] == nums[last]) {
				int left_min = self(self, nums, left, mid);
				int right_min =
					self(self, nums, mid + 1, right);
				return std::min(left_min, right_min);
			} else {
				left = mid + 1;
			}
		}
		return nums[left];
	};
	return recursive_find_min(recursive_find_min, nums, 0, nums.size() - 1);
}

std::vector<int> Solution::constructRectangle(int area)
{
	int w = static_cast<int>(std::sqrt(area));
	while (area % w != 0) {
		w--;
	}
	return std::vector<int>{ area / w, w };
}

int Solution::countNodes(TreeNode *root)
{
	if (root == nullptr) {
		return 0;
	}
	auto curr = root;
	int height = 0;
	while (curr->left != nullptr) {
		curr = curr->left;
		height++;
	}
	// height === level
	auto contains_nth_element = [](TreeNode *root, int height,
				       int n) -> bool {
		while (root != nullptr && 0 <= --height) {
			if (0 == ((1 << (height)) & n)) {
				root = root->left;
			} else {
				root = root->right;
			}
		}
		return root != nullptr;
	};

	int left = (1 << height);
	int right = (1 << (height + 1)) - 1;
	while (left < right) {
		auto mid = right - (right - left) / 2;
		if (contains_nth_element(root, height, mid)) {
			left = mid;
		} else {
			right = mid - 1;
		}
	}
	return left;
}

int Solution::minimumMoves(std::vector<int> &arr)
{
	return 0;
}

int Solution::uniquePathsIII(std::vector<std::vector<int> > &grid)
{
	return 0;
}

int Solution::firstMissingPositive(std::vector<int> &nums)
{
	int length = static_cast<int>(nums.size());
	nums.push_back(nums.back()); // dummy num
	for (int i = 0; i < length; i++) {
		if (nums[i] <= 0 || length + 1 <= nums[i] ||
		    nums[i] == nums[nums[i]]) {
			continue;
		}
		if (nums[i] <= i) {
			nums[nums[i]] = nums[i];
		} else {
			std::swap(nums[i], nums[nums[i]]);
			--i;
		}
	}
	for (int i = 1; i < length + 1; ++i) {
		if (nums[i] != i) {
			return i;
		}
	}
	return length + 1;
}

std::vector<std::vector<int> >
Solution::allPathsSourceTarget(std::vector<std::vector<int> > &graph)
{
	std::vector<std::vector<int> > ret;
	auto dfs = [&](auto &&dfs, int node, int target,
		       std::vector<int> &path) -> void {
		if (node == target) {
			ret.emplace_back(path);
		}
		for (auto next : graph[node]) {
			path.push_back(next);
			dfs(dfs, next, target, path);
			path.pop_back();
		}
	};
	std::vector<int> path{ 0 };
	path.reserve(graph.size());
	dfs(dfs, 0, static_cast<int>(graph.size() - 1), path);
	return ret;
}

std::vector<std::vector<std::string> >
Solution::accountsMerge(std::vector<std::vector<std::string> > &accounts)
{
	return std::vector<std::vector<std::string> >();
}

bool Solution::canReach(std::vector<int> &arr, int start)
{
	if (arr[start] == 0) {
		return true;
	}
	std::vector<bool> can_reach(arr.size(), false);
	std::vector<bool> visited(arr.size(), false);
	for (size_t i = 0; i < arr.size(); ++i) {
		if (arr[i] == 0) {
			can_reach[i] = true;
			visited[i] = true;
		}
	}
	auto dfs = [&](auto &&self, int index) -> bool {
		if (visited[index]) {
			return can_reach[index];
		}
		visited[index] = true;
		bool ret = false;
		if (0 <= index - arr[index]) {
			ret |= self(self, index - arr[index]);
		}
		if (index + arr[index] < arr.size()) {
			ret |= self(self, index + arr[index]);
		}
		return ret;
	};
	return dfs(dfs, start);
}

int Solution::nthMagicalNumber(int n, int a, int b)
{
	long lcm_a_b = std::lcm(a, b);
	long l = 1;
	long r = LONG_MAX;
	while (l < r) {
		auto m = l + (r - l) / 2;
		auto count = m / a + m / b - m / lcm_a_b;
		if (count < n)
			l = m + 1;
		else
			r = m;
	}
	int modulo = static_cast<int>(1e9) + 7;
	return l % modulo;
}

std::vector<int>
Solution::findOrder(int numCourses,
		    std::vector<std::vector<int> > &prerequisites)
{
	std::vector<int> ret;
	std::vector<int> in_degree(numCourses);
	std::unordered_map<int, std::vector<int> > adjacent;

	for (const auto &prerequisite : prerequisites) {
		adjacent[prerequisite[1]].push_back(prerequisite[0]);
		++in_degree[prerequisite[0]];
	}

	std::queue<int> q;
	for (int i = 0; i < numCourses; i++) {
		if (in_degree[i] == 0) {
			q.push(i);
		}
	}

	while (!q.empty()) {
		auto node = q.front();
		ret.push_back(node);
		q.pop();

		if (adjacent.contains(node)) {
			for (auto neighbor : adjacent[node]) {
				--in_degree[neighbor];

				if (in_degree[neighbor] == 0) {
					q.push(neighbor);
				}
			}
		}
	}
	if (ret.size() < numCourses) {
		return std::vector<int>();
	}
	return ret;
}

bool Solution::canReach(std::string s, int minJump, int maxJump)
{
	assert(!s.empty());
	assert(minJump <= maxJump);
	if (s.back() == '1') {
		return false;
	}

	s[0] = '2';
	int covered = 0;
	int len = static_cast<int>(s.length());
	for (int i = 0; i < s.length(); i++) {
		if (s[i] == '2') {
			for (int j = std::max(i + minJump, covered);
			     j < std::min(i + maxJump + 1, len); j++) {
				if (s[j] == '0')
					s[j] = '2';
			}
			covered = i + maxJump;
			if (s.back() == '2') {
				return true;
			}
		}
	}
	return s.back() == '2';
}

int Solution::maximumGood(std::vector<std::vector<int> > &statements)
{
	return 0;
}

int Solution::findRadius(std::vector<int> &houses, std::vector<int> &heaters)
{
	return 0;
}

int Solution::largestRectangleArea(std::vector<int> &heights)
{
	return 0;
}

int Solution::widthOfBinaryTree(TreeNode *root)
{
	if (root == nullptr) {
		return 0;
	}

	size_t ret = 1;
	std::queue<std::pair<TreeNode *, size_t> > queue;
	queue.push({ root, 0 });

	while (!queue.empty()) {
		auto count = queue.size();
		size_t left = SIZE_MAX;
		size_t right = 0;
		while (0 != count--) {
			root = queue.front().first;
			auto index = queue.front().second;
			left = std::min(left, index);
			right = std::max(right, index);
			queue.pop();
			if (root->left) {
				queue.push({ root->left, index * 2 + 1 });
			}
			if (root->right) {
				queue.push({ root->right, index * 2 + 2 });
			}
		}
		if (left != SIZE_MAX) {
			ret = std::max(right - left + 1, ret);
		}
	}
	return static_cast<int>(ret);
}

std::vector<std::string> Solution::summaryRanges(std::vector<int> &nums)
{
	std::vector<std::string> ret;
	if (nums.empty()) {
		return ret;
	}
	std::string tmp;
	for (size_t i = 0; i + 1 < nums.size(); i++) {
		if (tmp.empty()) {
			tmp = std::to_string(nums[i]);
		}
		if (nums[i + 1] == nums[i] + 1) {
			continue;
		} else if (nums[i] != std::stoi(tmp)) {
			tmp.append("->");
			tmp.append(std::to_string(nums[i]));
		}
		ret.emplace_back(tmp);
		tmp.clear();
	}
	if (tmp.empty()) {
		tmp = std::to_string(nums.back());
	} else {
		tmp.append("->");
		tmp.append(std::to_string(nums.back()));
	}
	ret.emplace_back(tmp);
	return ret;
}

int Solution::getKth(int lo, int hi, int k)
{
	std::unordered_map<int, int> hash_map;
	auto calculate_power = [&](auto &&self, int x) -> int {
		if (x == 1) {
			return 0;
		}
		if (hash_map.find(x) != hash_map.end()) {
			return hash_map[x];
		}
		int y = x;
		if ((x & 0x1) == 0) {
			x >>= 1;
		} else {
			x = 3 * x + 1;
		}
		hash_map[y] = 1 + self(self, x);
		return hash_map[y];
	};
	std::priority_queue<std::pair<int, int> > heap;
	for (auto i = lo; i <= hi; i++) {
		heap.push({ calculate_power(calculate_power, i), i });
		if (k < heap.size()) {
			heap.pop();
		}
	}
	return heap.top().second;
}

std::string Solution::minRemoveToMakeValid(std::string s)
{
	std::stack<char> stk;
	std::string ret;
	for (auto c : s) {
		if (c == '(') {
			stk.push('(');
			ret.push_back(c);
		} else if (c == ')') {
			if (!stk.empty()) {
				ret.push_back(c);
				stk.pop();
			}
		} else {
			ret.push_back(c);
		}
	}
	if (!stk.empty()) {
		if (ret.size() == stk.size()) {
			return "";
		}
		size_t j = 0;
		size_t i = ret.size();
		for (; 0 < --i && j < stk.size();) {
			if (ret[i] == '(') {
				j++;
			}
		}
		std::string tmp;
		for (size_t k = 0; k < ret.size(); k++) {
			if (k <= i || ret[k] != '(') {
				tmp.push_back(ret[k]);
			}
		}
		ret = std::move(tmp);
	}

	return ret;
}

int Solution::scoreOfParentheses(std::string s)
{
	return 0;
}

int Solution::minDominoRotations(std::vector<int> &tops,
				 std::vector<int> &bottoms)
{
	return 0;
}

std::vector<int> Solution::busiestServers(int k, std::vector<int> &arrival,
					  std::vector<int> &load)
{
	std::vector<int> count(k, 0);
	std::set<int> free_set;
	for (int i = 0; i < k; i++) {
		free_set.insert(i);
	}
	std::priority_queue<std::pair<int, int>,
			    std::vector<std::pair<int, int> >,
			    std::greater<std::pair<int, int> > >
		busy_heap; // {finish_time, index}
	for (size_t i = 0; i < arrival.size(); i++) {
		while (!busy_heap.empty() &&
		       busy_heap.top().first <= arrival[i]) {
			free_set.insert(busy_heap.top().second);
			busy_heap.pop();
		}
		if (free_set.empty()) {
			continue;
		} else {
			auto it = free_set.lower_bound(i % k);
			if (it == free_set.end()) {
				it = free_set.begin();
			}
			auto index = *it;
			free_set.erase(index);
			busy_heap.push({ arrival[i] + load[i], index });
			count[index]++;
		}
	}
	std::map<int, std::vector<int> > map;
	for (int i = 0; i < k; i++) {
		map[count[i]].push_back(i);
	}
	return map.rbegin()->second;
}

bool Solution::hasAllCodes(std::string s, int k)
{
	int need = 1 << k;
	std::vector<bool> got(need, false);
	int all_one = need - 1;
	int hash_val = 0;

	for (size_t i = 0; i < s.length(); i++) {
		// calculate hash for  s.substr(i - k + 1, i + 1)
		hash_val = ((hash_val << 1) & all_one) | (s[i] - '0');
		// hash only avalable when i - k + 1 > 0
		if (k - 1 <= i && !got[hash_val]) {
			got[hash_val] = true;
			need--;
			if (need == 0) {
				return true;
			}
		}
	}
	return false;
}

std::vector<int>
Solution::countRectangles(std::vector<std::vector<int> > &rectangles,
			  std::vector<std::vector<int> > &points)
{
	return std::vector<int>();
}

int Solution::consecutiveNumbersSum(int n)
{
	int ret = 1;
	int bound = 2 * n;

	auto is_k_consecutive = [](int n, int k) {
		if (k % 2 == 1) {
			return n % k == 0;
		} else {
			return n % k != 0 && 2 * n % k == 0;
		}
	};
	for (int i = 2; i * (i + 1) <= bound; i++) {
		if (is_k_consecutive(n, i)) {
			ret++;
		}
	}
	return ret;
}

std::vector<int>
Solution::fullBloomFlowers(std::vector<std::vector<int> > &flowers,
			   std::vector<int> &persons)
{
	return std::vector<int>();
}

int Solution::maxLength(std::vector<int> &nums)
{
	std::array<std::vector<int>, 11> factors = { std::vector<int>{},
						     {},
						     { 2 },
						     { 3 },
						     { 2 },
						     { 5 },
						     { 2, 3 },
						     { 7 },
						     { 2 },
						     { 3 },
						     { 2, 5 } };
	//std::vector<int> factors[12] = { {},	{},    { 2 },	 { 3 },
	//				 { 2 }, { 5 }, { 2, 3 }, { 7 },
	//				 { 2 }, { 3 }, { 2, 5 } };
	std::set<int> factor_set;
	size_t ret{ 2 };
	for (size_t i = 0, j = 0; i < nums.size(); i++) {
		for (bool flag = false;; j++, flag = false) {
			for (auto factor : factors[nums[i]]) {
				if (factor_set.contains(factor)) {
					flag = true;
					break;
				}
			}
			if (!flag) {
				break;
			}
			for (auto factor : factors[nums[j]]) {
				if (factor_set.contains(factor)) {
					factor_set.erase(factor);
				}
			}
		}
		for (auto factor : factors[nums[i]]) {
			factor_set.insert(factor);
		}
		ret = std::max(ret, i - j + 1);
	}
	return static_cast<int>(ret);
}

int Solution::maximumSum(std::vector<int> &nums)
{
	std::unordered_map<int, std::priority_queue<int> > hash_map;
	for (auto num : nums) {
		auto sum_of_digits = [](int num) -> int {
			int ret{ 0 };
			while (0 != num) {
				ret += num % 10;
				num /= 10;
			}
			return ret;
		};
		hash_map[sum_of_digits(num)].push(num);
	}
	if (hash_map.size() == nums.size()) {
		return -1;
	}
	int ret{ 0 };
	for (auto &&[k, v] : hash_map) {
		if (1 < v.size()) {
			auto tmp = v.top();
			v.pop();
			tmp += v.top();
			ret = std::max(ret, tmp);
		}
	}
	return ret;
}

int Solution::numTilePossibilities(std::string tiles)
{
	std::array<int, 26> counter = { 0 };
	for (char tile : tiles) {
		counter[tile - 'A']++;
	}

	std::function<int()> backtrack = [&]() -> int {
		int count = 0;
		for (int i = 0; i < 26; ++i) {
			if (0 < counter[i]) {
				counter[i]--; // Use one tile
				count +=
					1 +
					backtrack(); // Count this sequence + further sequences
				counter[i]++; // Backtrack
			}
		}
		return count;
	};

	return backtrack();
}

/* You are given a string word and a non-negative integer k.
Return the total number of of word that contain every vowel('a', 'e', 'i', 'o', and 'u') at least once and exactly k consonants. 
*/
long long Solution::countOfSubstrings(std::string word, int k)
{
	std::unordered_map<char, int> vowel_count;
	const std::unordered_set<char> vowels{ 'a', 'e', 'i', 'o', 'u' };
	int consonants{ 0 };
	long long ret{ 0 };
	for (size_t i = 0, j = 0; i < word.size(); i++) {
		if (consonants <= k and vowel_count.size() <= vowels.size()) {
			if (vowels.contains(word[i])) {
				vowel_count[word[i]]++;
				i++;
			} else if (consonants < k) {
				consonants++;
			} else if (vowel_count.size() < vowels.size()) {
				while (not vowels.contains(word[j])) {
					j++;
				}
				break;
			}
		}
		while (j < i and consonants == k and
		       vowel_count.size() == vowels.size()) {
			ret++;
			if (vowels.contains(word[j])) {
				vowel_count[word[j]]--;
				if (vowel_count[word[j]] == 0) {
					vowel_count.erase(word[j]);
				}
			} else {
				consonants--;
			}
			j++;
		}
	}

return ret;
}

int Solution::numberOfSubstrings(std::string s)
{
	int ret{ 0 };
	std::array<int, 3> last_pos{ -1, -1, -1 };
	for (int i = 0; i < s.length(); i++) {
		last_pos[s[i] - 'a'] = i;
		ret += 1 + *std::min_element(last_pos.cbegin(), last_pos.cend());
	}
	return ret;
}
