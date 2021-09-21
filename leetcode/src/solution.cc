#include "solution.h"
#include <queue>

#include "y_combinator.h"
#include <cassert>
#include <iostream>

#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#include <cstdlib>
#include <unordered_set>
#include <array>
int Solution::twoCitySchedCost(std::vector<std::vector<int> > &costs)
{
	int res = 0;
#if false
	std::sort(costs.begin(), costs.end(), [](std::vector<int> a, std::vector<int> b) { return a[0] - a[1] < b[0] - b[1]; });
	int n = costs.size() >> 1;
	for (int i = 0; i < n; i++)
		res += costs[i][0] + costs[2 * n - i - 1][1];
#else //noob version
	int count = costs.size();
	int sum[2] = { 0 };
	std::vector<int> minus;
	std::multimap<unsigned int, std::vector<int>, std::greater<int> > mmap;
	for (auto pair : costs) {
		mmap.insert({ abs(pair[0] - pair[1]), pair });
	}
	int i = 0, j = 0;
	for (auto &&e : mmap) {
		if ((count >> 1) <= i)
			res += e.second[1];
		else if ((count >> 1) <= j)
			res += e.second[0];
		else
			res += e.second[0] < e.second[1] ? (i++, e.second[0]) :
								 (j++, e.second[1]);
	}
	//int ans = 118 + 259 + 54 + 667 + 184 + 577;
#endif
	return res;
}

std::vector<std::vector<int> >
Solution::reconstructQueue(std::vector<std::vector<int> > &people)
{
	std::vector<std::vector<int> > res;
	int k = 0;
	std::sort(people.begin(), people.end(),
		  [](const std::vector<int> &lhs, const std::vector<int> &rhs) {
			  if (lhs.back() != rhs.back())
				  return lhs.back() < rhs.back();
			  return lhs.front() > rhs.front();
		  });
	size_t i = 0;
	while (i < people.size()) {
		int j = 0;
		k = people.at(i).back();
		while (k) {
			if (people.at(i).front() <= res.at(j).front())
				k--;
			j++;
		}
		k = people.at(i).back();
		res.insert(res.begin() + j, people.at(i));
		i++;
	}
	return res;
	//std::vector<std::vector<int>> res;
	//int k = 0;
	//std::sort(people.begin(), people.end(), [](std::vector<int> A, std::vector<int> B) { if (A.back() != B.back()) return A.back() < B.back(); return A.front() < B.front(); });
	//int i = 0;
	//while (i < people.size())
	//{
	//    if (people.at(i).back() == k) {
	//        res.push_back(people.at(i));
	//        i++;
	//    }
	//    else {
	//        int j = 0;
	//        k = people.at(i).back();
	//        while (k) {
	//            if (people.at(i).front() <= res.at(j).front())
	//                k--;
	//            j++;
	//        }
	//        k = people.at(i).back();
	//        res.insert(res.begin() + j, people.at(i));
	//        i++;
	//    }
	//}
	//return res;
}

int Solution::change(int amount, std::vector<int> coins)
{
	if (amount < coins[0])
		return 0;
	if (amount == coins[0])
		return 1;
	return 0;
}

bool Solution::isPowerOfTwo(int n)
{
	return n > 0 && (n & (n - 1)) == 0;
	if (n < 0)
		return false;
	int temp = n;
	int bits = 0;
	while (temp >>= 1)
		bits++;
	return (1 << bits) == n;
}

bool Solution::isPowerOfFour(int num)
{
	return num > 0 && (num & (num - 1)) == 0 and (num & 0xaaaaaaaa) == 0;
	if (num < 0)
		return false;
	int count[2] = { 0 };
	while (num) {
		count[num & 0x1]++;
		num >>= 1;
	}
	return count[1] == 1 && count[0] % 2 == 0;
}

bool Solution::isSubsequence(std::string s, std::string t)
{
	if (t.length() < s.length())
		return false;
	for (auto it = s.rbegin(); it != s.rend(); it++) {
		for (int i = t.length() - 1; i >= 0; i--) {
			if (t.at(i) == *it) {
				t.erase(i);
				break;
			}
			if (0 == i)
				return false;
		}
	}
	return true;
}

void Solution::sortColors(std::vector<int> &nums)
{
	int left = 0;
	int right = nums.size() - 1;
	int i = 0;
	while (left < right) {
		if (nums[i] == 0) {
			nums[left] ^= nums[i];
			nums[i] ^= nums[left];
			nums[left] ^= nums[i];
			left++;
		} else if (nums[i] == 2) {
			nums[right] ^= nums[i];
			nums[i] ^= nums[right];
			nums[right] ^= nums[i];
			right--;
		}
		i++;
	}
}

std::vector<int> Solution::largestDivisibleSubset(std::vector<int> &nums)
{
	std::vector<int> res;
	return res;
}

int Solution::findCheapestPrice(int n, std::vector<std::vector<int> > &flights,
				int src, int dst, int K)
{
	int res = 0;
	return res;
}

// TODO:
std::string Solution::validIPAddress(std::string IP)
{
	enum IPAddressType {
		IPv4,
		IPv6,
		Neither,
	};
	std::vector<std::string> strIPAddressType = { "IPv4", "IPv6",
						      "Neither" };
	// IPV4
	if (7 <= IP.size() && IP.size() <= 15) {
		return strIPAddressType[IPAddressType::IPv4];
	}
	// IPV6
	if (15 <= IP.size() && IP.size() <= 39) {
		return strIPAddressType[IPAddressType::IPv6];
	}

	return strIPAddressType[IPAddressType::Neither];
}
void Solution::solve(std::vector<std::vector<char> > &board)
{
	auto dfs = make_y_combinator([](auto &&dfs, int row, int column,
					std::vector<std::vector<char> > grid) {
		if (row < 0 || row >= grid.size() || column < 0 ||
		    column >= grid[0].size() || grid[row][column] != 'O')
			return;
		grid[row][column] -= 9;
		dfs(row - 1, column, grid);
		dfs(row + 1, column, grid);
		dfs(row, column - 1, grid);
		dfs(row, column + 1, grid);
	});
	for (size_t i = 0; i < board.size(); i++)
		for (size_t j = 0; j < board[0].size(); j++)
			if (i == 0 || i == board.size() - 1 || j == 0 ||
			    j == board[0].size() - 1)
				if (board[i][j] == 'O')
					dfs(i, j, board);

	for (size_t i = 0; i < board.size(); i++)
		for (size_t j = 0; j < board[0].size(); j++)
			board[i][j] += 9;
}

int Solution::hIndex(std::vector<int> &citations)
{
	// sort
	std::sort(citations.begin(), citations.end(), std::greater<int>());

	// binary search
	int left = 0, right = citations.size();
	while (left < right) {
		auto mid = left + (right - left) / 2;
		if (mid + 1 - citations[mid] <= 0)
			left = mid + 1;
		else
			right = mid;
	}
	return left;
}

std::string Solution::longestDupSubstring(std::string S)
{
	return std::string();
}

std::string Solution::getPermutation(int n, int k)
{
	k--;
	std::string res;
	std::vector<char> nums = { '1', '2', '3', '4', '5', '6', '7', '8', '9' };
	nums.resize(n);
	int count_of_last_permutation = 1;
	for (size_t i = 1; i < n; i++) {
		count_of_last_permutation *= i;
	}
	while (!nums.empty()) {
		int index = k / count_of_last_permutation;
		res.push_back(nums[index]);
		k %= count_of_last_permutation;
		count_of_last_permutation /=
			nums.size() - 1 ? nums.size() - 1 : 1;
		nums.erase(nums.begin() + index);
	}
	return res;
}

int Solution::calculateMinimumHP(std::vector<std::vector<int> > &dungeon)
{
	return 0;
}

// TODO:
int Solution::singleNumber(std::vector<int> &nums)
{
	int temps[2] = { 0 };
	int length = nums.size();
	for (size_t i = 0; i < length; i++) {
		if (1 == i % 2) {
			temps[0] ^= -nums[i];
			temps[1] ^= nums[i];
		} else {
			temps[0] ^= nums[i];
			temps[1] ^= -nums[i];
		}
	}
	//int ones = 0;
	//int twos = 0;
	//int not_threes = 0;
	//for (int n : nums) {
	//    twos |= (ones & n);
	//    ones ^= n;
	//    not_threes = ~(ones & twos);
	//    ones &= not_threes;
	//    twos &= not_threes;
	//}
	return temps[0];
	//return ones;
}

std::string Solution::longestDiverseString(int a, int b, int c)
{
	std::string res;
	return res;
}

std::vector<std::vector<int> >
Solution::merge(std::vector<std::vector<int> > &intervals)
{
	//intervals = { {2,3},{4,5},{6,7},{8,9},{1,10} };
	std::vector<std::vector<int> > res;
	sort(intervals.begin(), intervals.end());
	res.push_back(intervals[0]);
	for (int i = 1; i < intervals.size(); i++) {
		if (intervals[i][0] <= res.back()[1]) {
			res.back()[1] = res.back()[1] < intervals[i][1] ?
						      intervals[i][1] :
						      res.back()[1];
		} else {
			res.push_back(intervals[i]);
		}
	}
	return res;
}

int Solution::findDuplicate(std::vector<int> &nums)
{
	return 0;
}

// TODO : wrong
std::vector<std::string>
Solution::findItinerary(std::vector<std::vector<std::string> > &tickets)
{
	return std::vector<std::string>();
}

int Solution::uniquePaths(int m, int n)
{
	std::vector<std::vector<int> > dp(m, std::vector<int>(n, 0));

	for (size_t i = 0; i < m; i++) {
		dp[i][0] = 1;
	}
	for (size_t i = 0; i < n; i++) {
		dp[0][i] = 1;
	}
	for (size_t i = 1; i < m; i++) {
		for (size_t j = 1; j < n; j++) {
			dp[i][j] = dp[i][j - 1] + dp[i - 1][j];
		}
	}
	return dp[m - 1][n - 1];
}

int Solution::arrangeCoins(int n)
{
	long temp = static_cast<long>(n) * 8 + 1;
	return static_cast<int>((pow(temp, 0.5) - 1) / 2);
}

std::vector<int> Solution::prisonAfterNDays(std::vector<int> &cells, int N)
{
	if (0 == N)
		return cells;
	std::vector<std::vector<int> > results;
	results.push_back(cells);
	while (results.size() < 3 || results.back() != results[1]) {
		std::vector<int> temp(8, 0);
		for (size_t i = 1; i < 7; i++)
			temp[i] = results.back()[i - 1] ^
				  results.back()[i + 1] ^ 1;
		results.push_back(temp);
	}
	results[0] = results[results.size() - 2];
	return results[N % (results.size() - 2)];
}

int Solution::hammingDistance(int x, int y)
{
	int n = x ^ y;
	int res = 0;
	while (n) {
		if (n & 1)
			res++;
		n >>= 1;
	}
	return res;
}

std::vector<int> Solution::plusOne(std::vector<int> &digits)
{
	size_t length = digits.size();
	std::vector<int> res(length, 0);
	for (int i = length - 1; i >= 0; i--) {
		if (length - 1 == i || (9 == digits[i + 1] && 0 == res[i + 1]))
			res[i] = (digits[i] + 1) % 10;
		else
			res[i] = digits[i];
	}
	if (0 == res[0])
		res.insert(res.begin(), 1);
	return res;
}

bool Solution::isUgly(int num)
{
	if (num <= 0)
		return false;
	while (0 == num % 2) {
		num /= 2;
	}
	while (0 == num % 3) {
		num /= 3;
	}
	while (0 == num % 5) {
		num /= 5;
	}
	return 1 == num;
}

int Solution::nthUglyNumber(int n)
{
	int i2 = 0, i3 = 0, i5 = 0;
	int dp[1690] = { 1 };
	for (size_t i = 1; i < n; i++) {
		dp[i] = std::min(std::min(dp[i2] * 2, dp[i3] * 3), dp[i5] * 5);
		if (dp[i2] * 2 <= dp[i])
			i2++;
		if (dp[i3] * 3 <= dp[i])
			i3++;
		if (dp[i5] * 5 <= dp[i])
			i5++;
	}
	return dp[n - 1];
}

int Solution::islandPerimeter(std::vector<std::vector<int> > &grid)
{
	int res = 0;
	if (grid.empty() || grid[0].empty())
		return res;
	size_t m = grid.size(), n = grid[0].size();
	for (size_t i = 0; i < m; i++) {
		if (1 == grid[i][0])
			res++;
		if (1 == grid[i][n - 1])
			res++;
	}
	for (size_t i = 0; i < n; i++) {
		if (1 == grid[0][i])
			res++;
		if (1 == grid[m - 1][i])
			res++;
	}
	for (size_t i = 0; i < m - 1; i++) {
		for (size_t j = 0; j < n - 1; j++) {
			if (1 == grid[i][j] ^ grid[i][j + 1])
				res++;
			if (1 == grid[i][j] ^ grid[i + 1][j])
				res++;
		}
	}
	for (size_t i = 0; i < m - 1; i++) {
		if (1 == grid[i][n - 1] ^ grid[i + 1][n - 1])
			res++;
	}
	for (size_t i = 0; i < n - 1; i++) {
		if (1 == grid[m - 1][i] ^ grid[m - 1][i + 1])
			res++;
	}
	return res;
}

std::vector<std::vector<int> > Solution::threeSum(std::vector<int> &nums)
{
	//nums = { 0,0,0,0 };
	//nums = { -2,0,1,1,2 };
	std::vector<std::vector<int> > res;
	std::sort(nums.begin(), nums.end());

	for (size_t i = 0; i < nums.size(); i++) {
		if (0 < i && nums[i] == nums[i - 1])
			continue;
		if (0 < nums[i])
			break;
		size_t left = i + 1, right = nums.size() - 1;
		while (left < right) {
			if ((i + 1 < left && nums[left] == nums[left - 1]) ||
			    nums[left] + nums[right] < -nums[i]) {
				left++;
				continue;
			}
			if ((right < nums.size() - 1 &&
			     nums[right] == nums[right + 1]) ||
			    nums[left] + nums[right] > -nums[i]) {
				right--;
				continue;
			}
			if (nums[left] + nums[right] == -nums[i]) {
				res.push_back(
					{ nums[i], nums[left++], nums[right] });
			}
		}
	}
	return res;
}

std::vector<std::vector<int> > Solution::subsets(std::vector<int> &nums)
{
	std::vector<std::vector<int> > res;
	size_t size = nums.size();
	std::vector<bool> bitmask(size + 1, false);
	while (false == bitmask.back()) {
		std::vector<int> subset;
		for (size_t i = 0; i < size; i++) {
			if (bitmask[i])
				subset.push_back(nums[i]);
		}
		res.push_back(subset);
		size_t j = 0;
		bool carry = true;
		while (carry) {
			bitmask[j] = !bitmask[j];
			carry = !bitmask[j++];
		}
	}
	return res;
}

uint32_t Solution::reverseBits(uint32_t n)
{
	uint32_t res = 0;
	for (size_t i = 0; i < 32; i++) {
		res <<= 1;
		res |= (n & 1);
		n >>= 1;
	}
	return res;
}

double Solution::angleClock(int hour, int minutes)
{
	double minutes_angle = minutes * 6.0; /*360.0 / 60.0*/
	double hour_angle =
		(hour * 60.0 + minutes) * 0.5; /*360.0 / 12.0 / 60.0*/
	double res = 0.0 < minutes_angle - hour_angle ?
				   minutes_angle - hour_angle :
				   hour_angle - minutes_angle;
	return res < 180.0 ? res : 360.0 - res;
}

std::string Solution::reverseWords(std::string s)
{
	reverse(s.begin(), s.end());

	int n = s.size();
	int idx = 0;
	for (int start = 0; start < n; ++start) {
		if (s[start] != ' ') {
			if (idx != 0)
				s[idx++] = ' ';

			int end = start;
			while (end < n && s[end] != ' ')
				s[idx++] = s[end++];

			reverse(s.begin() + idx - (end - start),
				s.begin() + idx);

			start = end;
		}
	}
	s.erase(s.begin() + idx, s.end());
	return s;
}

char *Solution::reverseWords(char *s)
{
	return nullptr;
}

double Solution::myPow(double x, int n)
{
	if (0 == n)
		return 1.0;
	if (0.0 == x || 1.0 == x)
		return x;
	if (0 == n % 2 && 2 != n)
		return myPow(myPow(x, n >> 1), 2);
	if (n < 0)
		return myPow(1.0 / x, -n);
	return myPow(x, n - 1) * x;
}

int Solution::superPow(int a, std::vector<int> &b)
{
	return 0;
}

std::vector<int> Solution::topKFrequent(std::vector<int> &nums, int k)
{
	std::vector<int> res;
	std::unordered_map<int, int> hash;
	std::multimap<int, int, std::greater<int> > mmap;
	for (auto num : nums) {
		hash[num]++;
	}
	for (auto &&e : hash) {
		mmap.insert({ e.second, e.first });
	}
	for (auto it = mmap.begin(); k--; it++) {
		res.push_back((*it).second);
	}
	return res;
}

std::string Solution::addBinary(std::string a, std::string b)
{
	a = "1111", b = "1111";
	std::string res;
	if (a.length() < b.length()) {
		res = b;
		b = a;
		a = res;
	} else {
		res = a;
	}
	auto it = a.rbegin();
	int i = res.length() - 1;
	int carry = 0;
	for (auto jt = b.rbegin(); jt != b.rend(); it++, jt++, i--) {
		res[i] = *it + *jt - '0' + carry;
		carry = 0;
		if ('1' < res[i]) {
			res[i] -= 2;
			carry = 1;
		}
	}
	while (carry != 0 && it != a.rend()) {
		res[i] = *it + carry;
		carry = 0;
		if ('1' < res[i]) {
			res[i] -= 2;
			carry = 1;
		}
		it++;
		i--;
	}
	return 1 == carry ? "1" + res : res;
}

bool Solution::canFinish(int numCourses,
			 std::vector<std::vector<int> > &prerequisites)
{
	return false;
}

std::vector<int>
Solution::findOrder(int numCourses,
		    std::vector<std::vector<int> > &prerequisites)
{
	return std::vector<int>();
}

int Solution::scheduleCourse(std::vector<std::vector<int> > &courses)
{
	return 0;
}

ListNode *Solution::reverseBetween(ListNode *head, int m, int n)
{
	ListNode dummy;
	dummy.next = head;
	ListNode *prev = &dummy;

	while (--m != 0) {
		--n;
		prev = prev->next;
	}
	ListNode *con = prev;
	ListNode *curr = prev->next;
	while (n-- != 0) {
		auto next = curr->next;
		curr->next = prev;
		prev = curr;
		curr = next;
	}
	con->next->next = curr;
	con->next = prev;
	return dummy.next;
}

bool Solution::exist(std::vector<std::vector<char> > &board, std::string word)
{
	return false;
}

std::string Solution::convert(std::string s, int numRows)
{
#if false // Sort by Row
	int n = (numRows << 1) - 2;
	if (n < 2)
		return s;
	std::vector<std::string> rows(numRows, std::string());
	for (size_t i = 0; i < s.length(); i++) {
		if (numRows <= i % n)
			rows[n - i % n].push_back(s[i]);
		else
			rows[i % n].push_back(s[i]);
	}
	std::string result;
	for (const auto& row : rows)
		result.append(row);
	return result;
#else
	// Visit by Row
	std::string result;
	int row = 0;
	int n = (numRows << 1) - 2;

	for (size_t row = 0; row < numRows; row++) {
		// first and last row
		if (0 == row || numRows - 1 == row) {
			for (size_t i = 0; i * n + row < s.length(); i++)
				result.push_back(s[i * n + row]);
			continue;
		}
		// inner rows
		for (size_t i = 0;; i++) {
			if (i * n + row < s.length())
				result.push_back(s[i * n + row]);
			else
				break;
			if ((i + 1) * n - row < s.length())
				result.push_back(s[(i + 1) * n - row]);
			else
				break;
		}
	}
	return result;
#endif
}

char *Solution::convert(char *s, int numRows)
{
	return nullptr;
}

int Solution::numWaterBottles(int numBottles, int numExchange)
{
	return 0 == numBottles % (numExchange - 1) ?
			     numBottles + numBottles / (numExchange - 1) - 1 :
			     numBottles + numBottles / (numExchange - 1);
}

std::vector<int> Solution::singleNumbers(std::vector<int> &nums)
{
	return std::vector<int>();
}

int *Solution::singleNumber(int *nums, int numsSize, int *returnSize)
{
	return nullptr;
}

std::vector<std::vector<int> >
Solution::allPathsSourceTarget(std::vector<std::vector<int> > &graph)
{
	return std::vector<std::vector<int> >();
}

int Solution::findMin(std::vector<int> &nums)
{
	size_t left = 0;
	size_t right = nums.size() - 1;
	size_t idx = (left + right) >> 1;
	while (left < right) {
		if (nums[idx] < nums[right])
			right = idx;
		else if (nums[right] < nums[idx])
			left = idx + 1;
		else
			right--;
		idx = (left + right) >> 1;
	}
	return nums[right];
}

TreeNode *Solution::buildTree(std::vector<int> &inorder,
			      std::vector<int> &postorder)
{
	if (inorder.empty())
		return nullptr;
	int llength = -1;
	while (inorder.at(llength++) != postorder.back())
		;
	std::vector<int> linorder(inorder.begin(), inorder.begin() + llength);
	std::vector<int> rinorder(inorder.begin() + llength + 1, inorder.end());
	std::vector<int> lpostorder(postorder.begin(),
				    postorder.begin() + llength);
	std::vector<int> rpostorder(postorder.begin() + llength,
				    postorder.end() - 1);
	TreeNode *root = new TreeNode(postorder.back());
	root->left = buildTree(linorder, lpostorder);
	root->right = buildTree(rinorder, rpostorder);
	return root;
}

int Solution::leastInterval(std::vector<char> &tasks, int n)
{
	if (0 == n)
		return tasks.size();
	std::unordered_map<char, int> hash;
	int res = 0;
	int a = 0, b = 0;
	for (auto &&task : tasks)
		hash[task]++;
	for (auto &&e : hash)
		if (a < e.second)
			a = e.second, b = 1;
		else if (a == e.second)
			b++;
	res = (n + 1) * (a - 1) + b;
	return res < tasks.size() ? tasks.size() : res;
}

std::string Solution::reorganizeString(std::string S)
{
	return std::string();
}

void Solution::merge(std::vector<int> &nums1, int m, std::vector<int> &nums2,
		     int n)
{
	while (0 < m && 0 < n) {
		if (nums1[m - 1] < nums2[n - 1])
			nums1[m + n] = nums2[--n];
		else
			nums1[m + n] = nums1[--m];
	}
	while (0 < n) {
		nums1[n] = nums2[--n];
	}
}

std::vector<int> Solution::sortedSquares(std::vector<int> &A)
{
	std::vector<int> res(A.size());
	unsigned left = 0, right = A.size() - 1;
	for (int i = A.size() - 1; i >= 0; i--) {
		if (-A[left] < A[right]) {
			res[i] = A[right] * A[right];
			right--;
		} else {
			res[i] = A[left] * A[left];
			left++;
		}
	}
	return res;
}

ListNode *Solution::mergeTwoLists(ListNode *l1, ListNode *l2)
{
#if recursion
	if (!l1)
		return l2;
	if (!l2)
		return l1;
	if (l1->val < l2->val) {
		l1->next = mergeTwoLists(l1->next, l2);
		return l1;
	} else {
		l2->next = mergeTwoLists(l1, l2->next);
		return l2;
	}
#else
	ListNode sentinel;
	ListNode *curr = &sentinel;
	while (l1 && l2) {
		if (l1->val < l2->val) {
			curr->next = l1;
			l1 = l1->next;
		} else {
			curr->next = l2;
			l2 = l2->next;
		}
		curr = curr->next;
	}
	if (l1)
		curr->next = l1;
	else
		curr->next = l2;
	return sentinel.next;
#endif // recursive
}

ListNode *Solution::mergeKLists(std::vector<ListNode *> &lists)
{
	// or use mergeTwoLists
	ListNode sentinel;
	ListNode *curr = &sentinel;
	while (!lists.empty()) {
		int min = INT_MAX;
		for (size_t i = 0; i < lists.size(); i++) {
			if (!lists[i]) {
				lists.erase(lists.begin() + i);
				i--;
			} else if (lists[i]->val < min) {
				min = lists[i]->val;
			}
		}
		for (size_t i = 0; i < lists.size(); i++) {
			if (lists[i] && lists[i]->val == min) {
				curr->next = lists[i];
				lists[i] = lists[i]->next;
				break;
			}
		}
		curr = curr->next;
	}
	return sentinel.next;
}

std::vector<std::string> Solution::wordBreak(std::string s,
					     std::vector<std::string> &wordDict)
{
	std::vector<std::string> res;
	return res;
}

int Solution::integerBreak(int n)
{
	if (n < 4)
		return n - 1;
	int res = 1;
	while (4 < n) {
		n -= 3;
		res *= 3;
	}
	return res * n;
}

int Solution::climbStairs(int n)
{
	std::vector<int> dp(n + 1, 1);
	for (size_t i = 2; i < n + 1; i++) {
		dp[i] = dp[i - 1] + dp[i - 2];
	}
	return dp[n];
}

std::string Solution::getHint(std::string secret, std::string guess)
{
	int bulls_count = 0, cows_count = 0;
	for (int i = secret.length() - 1; i >= 0; i--) {
		if (secret[i] == guess[i]) {
			bulls_count++;
			secret.erase(secret.begin() + i);
			guess.erase(guess.begin() + i);
		}
	}
	std::unordered_map<char, int> map_secret, map_guess;
	for (auto &&chr : secret)
		map_secret[chr]++;
	for (auto &&chr : guess)
		map_guess[chr]++;
	for (auto &&e : map_secret)
		if (map_guess[e.first])
			cows_count += e.second < map_guess[e.first] ?
						    e.second :
						    map_guess[e.first];
	std::string hint;
	return std::to_string(bulls_count) + "A" + std::to_string(cows_count) +
	       "B";
}

bool Solution::detectCapitalUse(std::string word)
{
	if ('Z' < word[0]) {
		for (auto &&letter : word)
			if (letter <= 'Z')
				return false;
	} else {
		for (size_t i = 1; i < word.length() - 1; i++) {
			if ((word[i] - '[') * (word[i + 1] - '[') < 0)
				return false;
		}
	}
	return true;
}

std::vector<int> Solution::smallestRange(std::vector<std::vector<int> > &nums)
{
	return std::vector<int>();
}

int Solution::countGoodTriplets(std::vector<int> &arr, int a, int b, int c)
{
	//std::sort(arr.begin(), arr.end());
	int count = 0;
	for (size_t i = 0; i < arr.size() - 2; i++)
		for (size_t j = i + 1; j < arr.size() - 1; j++)
			for (size_t k = j + 1; k < arr.size(); k++)
				if ((arr[i] - arr[j] <= a) &&
				    (-a <= arr[i] - arr[j]) &&
				    (arr[j] - arr[k] <= b) &&
				    (-b <= arr[j] - arr[k]) &&
				    (arr[i] - arr[k] <= c) &&
				    (-c <= arr[i] - arr[k]))
					count++;
	return count;
}

int Solution::getWinner(std::vector<int> &arr, int k)
{
	int winner = arr[0];
	int win_count = 0;
	for (size_t i = 1; i < arr.size(); i++) {
		if (arr[i] < winner) {
			win_count++;
		} else {
			win_count = 1;
			winner = arr[i];
		}
		if (win_count == k)
			return winner;
	}
	return winner;
}

int Solution::minSwaps(std::vector<std::vector<int> > &grid)
{
	int n = grid.size();
	int m = grid[0].size();

	if (n == 1)
		return 0;
	if (n == 2) {
		if (grid[0].back() == 0)
			return 0;
		if (grid[1].back() == 0)
			return 1;
		return -1;
	}
	int res = 0;
	size_t i = 0;
	bool flag = true;
	for (i = 0; i < n; i++) {
		flag = true;
		for (int j = m - 1; j >= m - n + 1; j--) {
			if (1 == grid[i][j]) {
				flag = false;
				break;
			}
		}
		if (flag) {
			res += i;
			break;
		}
	}
	if (!flag)
		return -1;
	grid.erase(grid.begin() + i);
	int subres = minSwaps(grid);
	if (subres != -1)
		return res + subres;
	return -1;
}

int Solution::maxSum(std::vector<int> &nums1, std::vector<int> &nums2)
{
	std::unordered_map<int, bool> umap1, umap2;
	for (auto &&num : nums1) {
		umap1[num] = true;
	}
	for (auto &&num : nums2) {
		umap2[num] = true;
	}
	std::vector<int> nums_common;
	for (auto &&e : umap1) {
		if (umap2[e.first] == true)
			nums_common.push_back(e.first);
	}
	return 0;
}

bool Solution::isPalindrome(std::string s)
{
	int n = s.size();
	int left = 0, right = n - 1;
	while (left < right) {
		while (left < right && !isalnum(s[left])) {
			++left;
		}
		while (left < right && !isalnum(s[right])) {
			--right;
		}
		if (left < right) {
			if (tolower(s[left]) != tolower(s[right])) {
				return false;
			}
			++left;
			--right;
		}
	}
	return true;
}

int Solution::rob(std::vector<int> nums)
{
#if false
	// House Robber I
	if (nums.empty())
		return 0;
	if (1 == nums.size())
		return nums[0];
	std::vector<int> dp(nums.size(), 0);
	dp[0] = nums[0];
	dp[1] = std::max(nums[0], nums[1]);
	for (size_t i = 2; i < nums.size(); i++)
	{
		dp[i] = std::max(dp[i - 2] + nums[i], dp[i - 1]);
	}
	return dp.back();
#else
	// House Robber II
	if (nums.empty())
		return 0;
	if (1 == nums.size())
		return nums[0];
	if (2 == nums.size())
		return std::max(nums[0], nums[1]);
	std::vector<std::vector<int> > dp(nums.size() - 1, { 0, 0 });
	std::vector<int> dp2(nums.size() - 1, 0);
	dp[0][0] = nums[0];
	dp[1][0] = std::max(nums[0], nums[1]);
	dp[0][1] = nums[1];
	dp[1][1] = std::max(nums[1], nums[2]);
	for (size_t i = 2; i < nums.size() - 1; i++) {
		dp[i][0] = std::max(dp[i - 2][0] + nums[i], dp[i - 1][0]);
		dp[i][1] = std::max(dp[i - 2][1] + nums[i + 1], dp[i - 1][1]);
	}
	return std::max(dp.back().at(0), dp.back().back());
#endif
}
int Solution::rob(TreeNode *root)
{
	static std::unordered_map<TreeNode *, int> robmap;
	if (!root)
		return 0;
	if (robmap[root])
		return robmap[root];
	int res = root->val;
	if (root->left)
		res += rob(root->left->left) + rob(root->left->right);
	if (root->right)
		res += rob(root->right->left) + rob(root->right->right);
	res = std::max(res, rob(root->left) + rob(root->right));
	robmap[root] = res;
	return res;
	return 0;
}
int Solution::diameterOfBinaryTree(TreeNode *root)
{
	int ans = 0;
	static auto height =
		make_y_combinator([&](auto &&height, TreeNode *node) -> int {
			if (node == nullptr)
				return -1;
			int L = height(node->left);
			int R = height(node->right);
			ans = std::max(ans, L + 1 + R + 1);
			return std::max(L, R) + 1;
		});
	height(root);
	return ans;
}

std::vector<int> Solution::findDuplicates(std::vector<int> &nums)
{
	std::vector<int> res;
	for (size_t i = 0; i < nums.size(); i++) {
		size_t index = 0 < nums[i] ? nums[i] - 1 : -nums[i] - 1;
		if (nums[index] < 0)
			res.push_back(index + 1);
		nums[index] *= -1;
	}
	return res;
}

std::vector<int> Solution::findDisappearedNumbers(std::vector<int> &nums)
{
	std::vector<int> res;
	for (size_t i = 0; i < nums.size(); i++) {
		size_t index = 0 < nums[i] ? nums[i] - 1 : -nums[i] - 1;
		if (nums[index] > 0)
			nums[index] *= -1;
	}
	for (size_t i = 0; i < nums.size(); i++) {
		if (0 < nums[i])
			res.push_back(i + 1);
	}
	return res;
}

std::vector<std::vector<int> > Solution::verticalOrder(TreeNode *root)
{
	std::vector<std::vector<int> > res;
	std::queue<std::pair<TreeNode *, int> > q;
	std::map<int, std::vector<int> > map;
	q.push(std::make_pair(root, 0));
	while (!q.empty()) {
		auto node = q.front().first;
		auto x = q.front().second;
		auto val = node->val;
		if (0 == map.count(x))
			map[x] = std::vector<int>();
		map[x].push_back(node->val);
		q.pop();
		if (node->left)
			q.push(std::make_pair(node->left, x - 1));
		if (node->right)
			q.push(std::make_pair(node->right, x + 1));
	}
	for (auto e : map) {
		res.push_back(e.second);
	}
	return res;
}

std::vector<std::vector<int> > Solution::verticalTraversal(TreeNode *root)
{
	std::vector<std::vector<int> > res;
	std::queue<std::pair<TreeNode *, std::pair<int, int> > > q;
	std::map<int, std::vector<std::pair<int, int> > > map;
	q.push(std::make_pair(root, std::make_pair(0, 0)));
	while (!q.empty()) {
		auto node = q.front().first;
		auto x = q.front().second.first;
		auto y = q.front().second.second;
		auto val = node->val;
		if (0 == map.count(x))
			map[x] = std::vector<std::pair<int, int> >();
		map[x].push_back(std::make_pair(node->val, y));
		q.pop();
		if (node->left)
			q.push(std::make_pair(node->left,
					      std::make_pair(x - 1, y + 1)));
		if (node->right)
			q.push(std::make_pair(node->right,
					      std::make_pair(x + 1, y + 1)));
	}
	for (auto &&e : map)
		std::sort(e.second.begin(), e.second.end(),
			  [](const std::pair<int, int> &lhs,
			     const std::pair<int, int> &rhs) {
				  if (lhs.second != rhs.second)
					  return lhs.second < rhs.second;
				  return lhs.first < rhs.first;
			  });
	for (auto &&e : map) {
		std::vector<int> temp;
		for (auto &&pair : e.second)
			temp.push_back(pair.first);
		res.push_back(temp);
	}
	return res;
}

std::vector<int> Solution::inorderTraversal(TreeNode *root)
{
	std::vector<int> res;
	TreeNode *predecessor = nullptr;
	TreeNode *curr = root;
	while (curr) {
		if (!curr->left) {
			res.push_back(curr->val);
			curr = curr->right;
		} else {
			predecessor = curr->left;
			while (predecessor->right && predecessor->right != curr)
				predecessor = predecessor->right;
			if (!predecessor->right) {
				predecessor->right = curr;
				//res.push_back(curr->val); // caution!!!
				curr = curr->left;
			} else {
				predecessor->right = nullptr;
				res.push_back(curr->val);
				curr = curr->right;
			}
		}
	}
	return res;
}

void Solution::recoverTree(TreeNode *root)
{
	TreeNode *x = nullptr;
	TreeNode *y = nullptr;
	TreeNode *prev = nullptr;
	TreeNode *curr = root;
	std::stack<TreeNode *> s;
	while (true) {
		while (curr) {
			s.push(curr);
			curr = curr->left;
		}
		if (s.empty())
			break;
		curr = s.top();
		s.pop();
		if (prev && curr->val < prev->val) {
			y = curr;
			if (!x)
				x = prev;
			else
				break;
		}
		prev = curr;
		curr = curr->right;
	}
	int temp = x->val;
	x->val = y->val;
	y->val = temp;
}

bool Solution::hasPathSum(TreeNode *root, int sum)
{
	bool left_has_path_sum = false, right_has_path_sum = false,
	     curr = false;
	if (root) {
		if (!root->left && !root->right)
			curr = (sum == root->val);
		if (root->left)
			left_has_path_sum =
				hasPathSum(root->left, sum - root->val);
		if (root->right)
			right_has_path_sum =
				hasPathSum(root->right, sum - root->val);
	}
	return curr || left_has_path_sum || right_has_path_sum;
}
#if ROOT_TO_LEAF
std::vector<std::vector<int> > Solution::pathSum(TreeNode *root, int sum)
{
	std::vector<std::vector<int> > ret;
	auto dfs = make_y_combinator([&](auto &&dfs, TreeNode *node,
					 int targetSum, std::vector<int> path) {
		if (node == nullptr)
			return;
		path.push_back(node->val);
		targetSum -= node->val;
		if (targetSum == 0 && node->left == nullptr &&
		    node->right == nullptr) {
			ret_.emplace_back(path);
		} else {
			dfs(node->left, targetSum, path);
			dfs(node->right, targetSum, path);
		}
	});
	dfs(root, targetSum, std::vector<int>{});
	return ret;
}
#else
// TODO:
int Solution::pathSum(TreeNode *root, int sum)
{
	static int first = sum;
	int left_path_sum = 0, right_path_sum = 0, curr = 0;
	if (root) {
		curr = sum == root->val ? 1 : 0;
		if (sum == first) {
			if (root->left)
				left_path_sum =
					pathSum(root->left, sum - root->val) +
					pathSum(root->left, sum);
			if (root->right)
				right_path_sum =
					pathSum(root->right, sum - root->val) +
					pathSum(root->right, sum);
		} else {
			if (root->left)
				left_path_sum =
					pathSum(root->left, sum - root->val);
			if (root->right)
				right_path_sum =
					pathSum(root->right, sum - root->val);
		}
	}
	if (root && root->val == 10)
		auto k = 0;
	if (root && root->val == 5)
		auto k = 0;
	if (root && root->val == 3)
		auto k = 0;
	return left_path_sum + right_path_sum + curr;
}
#endif // From root to leaf

int Solution::closestValue(TreeNode *root, double target)
{
	return 0;
}

int Solution::orangesRotting(std::vector<std::vector<int> > &grid)
{
	std::queue<std::pair<size_t, size_t> > q;
	int fresh = 0;
	for (size_t row = 0; row < grid.size(); row++)
		for (size_t column = 0; column < grid[0].size(); column++)
			if (grid[row][column] == 2)
				q.push(std::make_pair(row, column));
			else if (grid[row][column] == 1)
				fresh++;
	unsigned size = 0;
	int rot_times = 0;
	while (!q.empty() && fresh != 0) {
		rot_times++;
		size = q.size();
		while (size-- != 0) {
			auto rotted_coord = q.front();
			q.pop();
			std::vector<std::pair<size_t, size_t> > rotting_coords;
			if (0 < rotted_coord.first)
				rotting_coords.push_back(
					std::make_pair(rotted_coord.first - 1,
						       rotted_coord.second));
			if (rotted_coord.first + 1 < grid.size())
				rotting_coords.push_back(
					std::make_pair(rotted_coord.first + 1,
						       rotted_coord.second));
			if (0 < rotted_coord.second)
				rotting_coords.push_back(std::make_pair(
					rotted_coord.first,
					rotted_coord.second - 1));
			if (rotted_coord.second + 1 < grid[0].size())
				rotting_coords.push_back(std::make_pair(
					rotted_coord.first,
					rotted_coord.second + 1));
			for (auto coord : rotting_coords)
				if (1 == grid[coord.first][coord.second]) {
					fresh--;
					grid[coord.first][coord.second] = 2;
					q.push(std::make_pair(coord.first,
							      coord.second));
				}
		}
	}
	return fresh == 0 ? rot_times : -1;
}

int Solution::titleToNumber(std::string s)
{
	int res = 0;
	for (const auto &chr : s) {
		res *= 26;
		res += chr - 'A';
	}
	return res;
}

int Solution::countBinarySubstrings(std::string s)
{
	if (s.empty())
		return 0;
	int count[2] = { 0 };
	count[s[0] - '0'] = 1;
	char last = s[0];
	int ret = 0;
	for (size_t i = 1; i < s.length(); i++) {
		if (s[i] == last) {
			count[last - '0']++;
		} else {
			ret += std::min(count[0], count[1]);
			count[s[i] - '0'] = 1;
			last = s[i];
		}
	}
	ret += std::min(count[0], count[1]);
	return ret;
}

std::string Solution::makeGood(std::string s)
{
	bool is_great = false;
	while (!is_great) {
		is_great = true;
		for (int i = s.length() - 2; i >= 0; i--)
			if (tolower(s[i]) == tolower(s[i + 1]) &&
			    s[i] != s[i + 1]) {
				s.erase(s.begin() + i, s.begin() + i + 2);
				is_great = false;
				break;
			}
	}
	return s;
}

char Solution::findKthBit(int n, int k)
{
	int length = (1 << n) - 1;
	if (k == 1)
		return '0';
	if (length / 2 + 1 == k)
		return '1';
	if (length / 2 + 1 < k)
		return 97 - findKthBit(n - 1, (1 << n) - k);
	return findKthBit(n - 1, k);
}

// TODO:
int Solution::maxNonOverlapping(std::vector<int> &nums, int target)
{
	std::map<int, int> map;
	for (size_t i = 0; i < nums.size(); i++) {
	}
	return 0;
}

std::vector<std::vector<int> > Solution::generate(int numRows)
{
	std::vector<std::vector<int> > res;
	for (size_t i = 0; i < numRows; i++) {
		std::vector<int> row(i + 1, 1);
		for (size_t j = 1; j < i; j++)
			row[j] = res.back().at(j - 1) + res.back().at(j);
		res.push_back(row);
	}
	return res;
}

std::vector<int> Solution::getRow(int rowIndex)
{
	std::vector<int> res(rowIndex + 1, 1);
	int hi = rowIndex, lo = 1;
	int i = 0;
	while (lo < hi) {
		long temp = res[i++] * hi--;
		res[i] = temp / lo++;
		res[rowIndex - i - 1] = res[i];
	}
	return res;
}

Node *Solution::cloneGraph(Node *node)
{
	if (!node)
		return nullptr;
	static std::map<Node *, Node *> map;
	std::vector<Node *> neighbors;
	Node *res = new Node(node->val);
	map[node] = res;
	for (auto &&neighbor : node->neighbors) {
		if (map.count(neighbor) == 0)
			map[neighbor] = cloneGraph(neighbor);
		neighbors.push_back(map[neighbor]);
	}
	res->neighbors = neighbors;
	return res;
}

ListNode *Solution::addTwoNumbers(ListNode *l1, ListNode *l2)
{
	int sum = 0;
	ListNode *l3 = nullptr;
	ListNode **node = &l3;
	while (l1 != nullptr || l2 != nullptr || 0 < sum) {
		if (l1 != nullptr) {
			sum += l1->val;
			l1 = l1->next;
		}
		if (l2 != nullptr) {
			sum += l2->val;
			l2 = l2->next;
		}
		(*node) = new ListNode(sum % 10);
		sum /= 10;
		node = &((*node)->next);
	}
	return l3;
}

std::string Solution::addStrings(std::string num1, std::string num2)
{
	auto it = num1.rbegin();
	auto jt = num2.rbegin();
	std::string res;
	int carry = 0;
	while (0 != carry || it != num1.rend() || jt != num2.rend()) {
		auto temp = carry;
		if (it != num1.rend())
			temp += *it++ - '0';
		if (jt != num2.rend())
			temp += *jt++ - '0';
		carry = 9 < temp ? 1 : 0;
		res.push_back(temp % 10 + '0');
	}
	std::reverse(res.begin(), res.end());
	return res;
}

std::string Solution::multiply(std::string num1, std::string num2)
{
	for (size_t i = 0; i < num1.length(); i++) {
	}
	return std::string();
}

bool Solution::isValid(std::string s)
{
	int n = s.size();
	if ((n & n - 1) != 0)
		return false;

	std::unordered_map<char, char> pairs = { { ')', '(' },
						 { ']', '[' },
						 { '}', '{' } };
	std::stack<char> stk;
	for (char ch : s) {
		if (pairs.count(ch)) {
			if (stk.empty() || stk.top() != pairs[ch]) {
				return false;
			}
			stk.pop();
		} else {
			stk.push(ch);
		}
	}
	return stk.empty();
}

int Solution::longestPalindrome(std::string s)
{
	std::unordered_map<char, int> umap;
	for (const auto &ch : s)
		umap[ch]++;
	int res = 0;
	for (const auto &e : umap) {
		if (e.second % 2 == 0)
			res += e.second;
		else
			res += e.second - 1;
	}
	if (s.length() != res)
		res++;
	return res;
}

bool Solution::canPermutePalindrome(std::string s)
{
	std::unordered_map<char, bool> umap;
	for (const auto &ch : s)
		umap[ch] = !umap[ch];
	int count = 0;
	for (const auto &e : umap)
		if (e.second)
			count++;
	return count < 2;
}

int Solution::removeBoxes(std::vector<int> &boxes)
{
	return 0;
}

int Solution::eraseOverlapIntervals(std::vector<std::vector<int> > &intervals)
{
	std::sort(intervals.begin(), intervals.end());
	return 0;
}

std::vector<int> Solution::findPermutation(std::string s)
{
	return std::vector<int>();
}

bool Solution::threeConsecutiveOdds(std::vector<int> &arr)
{
	for (size_t i = 0; i + 2 < arr.size(); i++) {
		if ((arr[i] & 1) == 0)
			continue;
		if ((arr[i] & arr[i + 1] & arr[i + 2] & 1) == 1)
			return true;
	}
	return false;
}

int Solution::minOperations(int n)
{
	return n / 2 * (n - n / 2);
	return 0;
}

int Solution::maxDistance(std::vector<int> &position, int m)
{
	std::sort(position.begin(), position.end());
	std::vector<int> distance;
	for (size_t i = 0; i + 1 < position.size(); i++)
		distance.push_back(position.at(i + 1) - position.at(i));
	if (2 == m)
		return position.back() - position.at(0);
	if (3 == m)
		return std::min(position.at(position.size() - 2) -
					position.at(0),
				position.back() - position.at(1));
	return 0;
}

int Solution::minDays(int n)
{
	static std::unordered_map<int, int> dp;
	if (n <= 1)
		return n;
	if (dp.count(n) == 0)
		dp[n] = 1 + std::min(n % 2 + minDays(n / 2),
				     n % 3 + minDays(n / 3));
	return dp[n];
}

int Solution::maxProfit_1st(std::vector<int> &prices)
{
	int min_price = INT_MAX;
	int max_profit = 0;
	for (auto &&price : prices) {
		if (price < min_price)
			min_price = price;
		if (max_profit < price - min_price)
			max_profit = price - min_price;
	}
	return max_profit;
}

int Solution::maxProfit_2nd(std::vector<int> &prices)
{
	return 0;
}

// TODO:
int Solution::maxProfit_3rd(std::vector<int> &prices)
{
	std::vector<int> profits;
	auto profit_once = maxProfit_1st(prices);
	if (0 == profit_once)
		return 0;
	for (size_t i = 1; i + 1 < prices.size(); i++) {
		// max = maxbefore + max_post
		std::vector<int> prices_before(prices.begin(),
					       prices.begin() + i + 1);
		std::vector<int> prices_after(prices.begin() + i + 1,
					      prices.end());
		// max before
		//maxProfit_1st(prices_before);
		// max after
		//maxProfit_1st(prices_after);
		profits.push_back(maxProfit_1st(prices_before) +
				  maxProfit_1st(prices_after));
		// sort max before and after
		//add ttwo back()
	}
	int max_profit = 0;
	for (const auto &profit : profits)
		if (max_profit < profit)
			max_profit = profit;
	if (max_profit < profit_once)
		max_profit = profit_once;
	return max_profit;
}

// TODO:
int Solution::maxProfit_4th(int k, std::vector<int> &prices)
{
	size_t left = 0, right = 1;
	std::vector<int> profits;
	while (right < prices.size()) {
		if (prices[left] < prices[right]) {
			while (right + 1 < prices.size() &&
			       prices[right] <= prices[right + 1]) {
				right++;
			}
			profits.push_back(prices[right] - prices[left]);
			left = right;
			right++;
		} else {
			left = right;
			right++;
		}
	}
	std::sort(profits.rbegin(), profits.rend());
	int result = 0;
	for (size_t i = 0; i < profits.size() && i < k; i++) {
		result += profits[i];
	}
	return result;
}

int Solution::maxProfit_5th(std::vector<int> &prices)
{
	return 0;
}

std::vector<int> Solution::distributeCandies(int candies, int num_people)
{
	candies = 44;
	int row_count = 0;
	int square_of_num_people = num_people * num_people;
	int limit = 0;
	while (limit < candies) {
		row_count++;
		limit = row_count * num_people * (num_people + 1) / 2 +
			square_of_num_people * (row_count - 1) * row_count / 2;
	}
	int minus = limit - candies;
	std::vector<int> res(num_people, 0);
	size_t i = 0;
	for (; i < num_people; i++) {
		res[i] = row_count * (i + 1) +
			 num_people * (row_count - 1) * row_count / 2;
	}
	while (0 < minus) {
		i--;
		res[i] -= (row_count - 1) * num_people + i + 1;
		minus -= (row_count - 1) * num_people + i + 1;
	}
	res[i] -= minus;
	return res;
}

std::vector<int> Solution::numsSameConsecDiff(int N, int K)
{
	return std::vector<int>();
}

std::string Solution::toGoatLatin(std::string S)
{
	std::vector<std::string> word_list;
	int last_index = 0;
	for (size_t i = 0; i < S.length(); i++)
		if (' ' == S.at(i)) {
			word_list.push_back(
				S.substr(last_index, i - last_index));
			last_index = i + 1;
		}
	word_list.push_back(S.substr(last_index, S.length() - last_index));
	std::string res;
	std::string suffix = "maa";
	auto isVowel = [](char ch) {
		return ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' ||
		       ch == 'u' || ch == 'A' || ch == 'E' || ch == 'I' ||
		       ch == 'O' || ch == 'U';
	};
	for (size_t i = 0; i < word_list.size(); i++) {
		if (isVowel(word_list.at(i).at(0))) {
			res.append(word_list.at(i));
		} else {
			for (size_t j = 1; j < word_list.at(i).length(); j++)
				res.push_back(word_list.at(i).at(j));
			res.push_back(word_list.at(i).at(0));
		}
		res.append(suffix);
		suffix.push_back('a');
		if (i + 1 != word_list.size())
			res.push_back(' ');
	}
	return res;
}

void Solution::reorderList(ListNode *head)
{
	ListNode *fast = head, *slow = head;
	while (fast && fast->next) {
		slow = slow->next;
		fast = fast->next->next;
	}

	//reverse second half
	ListNode *prev = nullptr;
	ListNode *curr = slow;
	while (curr) {
		ListNode *next = curr->next;
		curr->next = prev;
		prev = curr;
		curr = next;
	}
	slow = prev;

	// merge
	while (slow && slow->next) {
		auto temp = head->next;
		head->next = slow;
		head = temp;
		temp = slow->next;
		slow->next = head;
		slow = temp;
	}
}

std::vector<std::vector<char> >
Solution::updateBoard(std::vector<std::vector<char> > &board,
		      std::vector<int> &click)
{
	int dir_x[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int dir_y[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

	auto dfs = make_y_combinator([&](auto &&dfs, int row, int column) {
		if (row < 0 || row >= board.size() || column < 0 ||
		    column >= board[0].size() || board[row][column] != 'E')
			return;
		else if ('E' == board[row][column]) {
			int count = 0;
			int row_size = board.size(),
			    column_size = board.at(0).size();
			for (size_t i = 0; i < 8; i++) {
				auto x = row + dir_x[i];
				auto y = column + dir_y[i];
				if (0 <= x && x < row_size && 0 <= y &&
				    y < column_size && 'M' == board[x][y])
					count++;
			}
			if (0 == count) {
				board[row][column] = 'B';
				dfs(row - 1, column - 1);
				dfs(row - 1, column);
				dfs(row - 1, column + 1);
				dfs(row, column - 1);
				dfs(row, column + 1);
				dfs(row + 1, column - 1);
				dfs(row + 1, column);
				dfs(row + 1, column + 1);
			} else {
				board[row][column] = count + '0';
			}
		}
	});
	if ('M' == board[click.at(0)][click.at(1)])
		board[click.at(0)][click.at(1)] = 'X';
	else
		dfs(click.at(0), click.at(1));
	return board;
}

int Solution::minDepth(TreeNode *root)
{
	if (root == nullptr)
		return 0;
	std::queue<TreeNode *> queue;
	queue.push(root);
	int res = 0;
	int count = 0;
	while (!queue.empty()) {
		res++;
		count = queue.size();
		while (0 != count--) {
			TreeNode *curr = queue.front();
			queue.pop();
			if (nullptr == curr->left && nullptr == curr->right)
				return res;
			if (curr->left)
				queue.push(curr->left);
			if (curr->right)
				queue.push(curr->right);
		}
	}
	return res;
}

int Solution::maxDepth(TreeNode *root)
{
	if (root == nullptr)
		return 0;
	std::queue<TreeNode *> queue;
	queue.push(root);
	int res = 0;
	int count = 0;
	while (!queue.empty()) {
		res++;
		count = queue.size();
		while (0 != count--) {
			TreeNode *curr = queue.front();
			queue.pop();
			if (curr->left)
				queue.push(curr->left);
			if (curr->right)
				queue.push(curr->right);
		}
	}
	return res;
}

std::vector<int> Solution::sortArrayByParity(std::vector<int> &A)
{
	int left = 0, right = A.size() - 1;
	while (left < right) {
		if (0 == A[left] % 2) {
			left++;
			continue;
		}
		if (1 == A[right] % 2) {
			right--;
			continue;
		}
		std::swap(A[left++], A[right--]);
	}
	return A;
}

bool Solution::isBalanced(TreeNode *root)
{
	auto helper =
		make_y_combinator([](auto &&helper, TreeNode *root) -> int {
			if (root == nullptr)
				return 0;
			int leftHeight = 0, rightHeight = 0;
			if (-1 == (leftHeight = helper(root->left)) ||
			    -1 == (rightHeight = helper(root->right)) ||
			    1 < abs(leftHeight - rightHeight))
				return -1;
			else
				return std::max(leftHeight, rightHeight) + 1;
			return 0;
		});
	return 0 <= helper(root);
	return false;
}
int Solution::numOfMinutes(int n, int headID, std::vector<int> &manager,
			   std::vector<int> &informTime)
{
	auto helper = make_y_combinator([](auto &&helper, Node *node) {
		if (node->neighbors.empty())
			return node->val;
		auto max_time = 0;
		for (size_t i = 0; i < node->neighbors.size(); i++)
			max_time =
				std::max(max_time, helper(node->neighbors[i]) +
							   node->val);
		return max_time;
	});
	std::vector<Node> employees;
	for (size_t i = 0; i < n; i++) {
		Node employee(informTime[i]);
		employees.push_back(employee);
	}
	for (size_t i = 0; i < n; i++)
		if (headID != i)
			employees[manager[i]].neighbors.push_back(
				&employees[i]);
	int res;
	res = helper(&employees[headID]);
	return res;
}

bool Solution::isCousins(TreeNode *root, int x, int y)
{
	if (root == nullptr)
		return 0;
	std::queue<TreeNode *> queue;
	queue.push(root);
	int count = 0;
	std::map<int, int> parent;
	std::vector<std::vector<int> > level_order;
	while (!queue.empty()) {
		count = queue.size();
		std::vector<int> level;
		while (0 != count--) {
			TreeNode *curr = queue.front();
			queue.pop();
			level.push_back(curr->val);
			if (curr->left) {
				parent[curr->left->val] = curr->val;
				queue.push(curr->left);
			}
			if (curr->right) {
				parent[curr->left->val] = curr->val;
				queue.push(curr->right);
			}
		}
		auto result1 = std::find(level.begin(), level.end(), x);
		auto result2 = std::find(level.begin(), level.end(), y);
		if (result1 != level.end() && result2 != level.end())
			return parent[x] != parent[y];
		else if (result1 != level.end() || result2 != level.end())
			return false;
		level_order.push_back(level);
	}
	return false;
}

bool Solution::judgePoint24(std::vector<int> &nums)
{
	return false;
}

int Solution::rangeBitwiseAnd(int m, int n)
{
	int shift = 0;
	while (m < n) {
		m >>= 1;
		n >>= 1;
		++shift;
	}
	return m << shift;
}

bool Solution::repeatedSubstringPattern(std::string s)
{
	int offset = 0;
	while (++offset < (s.length() >> 1) + 1) {
		if (s[offset] != s[0])
			continue;
		if (0 != s.length() % offset)
			continue;
		size_t i = 1;
		for (; i < s.length() - offset; i++)
			if (s[offset + i] != s[i])
				break;
		if (i == s.length() - offset)
			return true;
	}
	return false;
}

int Solution::sumOfLeftLeaves(TreeNode *root)
{
	int res = 0;
	if (root == nullptr)
		return 0;
	if (root->left != nullptr && root->left->left == nullptr &&
	    root->left->left == nullptr)
		res += root->left->val;
	return res + sumOfLeftLeaves(root->right);
}

// TODO:
std::vector<std::vector<int> >
Solution::findSubsequences(std::vector<int> &nums)
{
	int upperbound = 1 << nums.size();
	std::vector<std::vector<int> > result;
	for (size_t bitmask = 0; bitmask < upperbound; bitmask++) {
		if (0 == (bitmask & (bitmask - 1))) //only one element
			continue;
		std::vector<int> subsequence;
		for (int idx = nums.size() - 1; 0 <= idx; idx--) {
			if (1 == ((bitmask >> idx) & 1))
				subsequence.push_back(
					nums.at(nums.size() - 1 - idx));
		}
		bool flag = true;
		for (size_t i = 0; i + 1 < subsequence.size(); i++)
			if (subsequence[i + 1] < subsequence[i]) {
				flag = false;
				break;
			}
		if (flag)
			result.push_back(subsequence);
	}
	std::sort(result.begin(), result.end());
	auto iter = std::unique(result.begin(), result.end());
	result.erase(iter, result.end());
	return result;
}

int Solution::mincostTickets(std::vector<int> &days, std::vector<int> &costs)
{
	std::vector<int> dp(days.size(), 0);
	dp[0] = std::min(std::min(costs[0], costs[1]), costs[2]);
	for (size_t i = 1; i < days.size(); i++) {
		std::vector<int> prices(3, dp[i - 1]);
		// use  1-day pass
		prices[0] = dp[i - 1] + costs[0];
		// use  7-day pass
		if (days[i] - days[0] < 7)
			prices[1] = costs[1];
		else {
			int last = i;
			while (days[i] - 7 < days[last])
				last--;
			prices[1] = costs[1] + dp[last];
		}
		// use  30-day pass
		if (days[i] - days[0] < 30)
			prices[2] = costs[2];
		else {
			int last = i;
			while (days[i] - 30 < days[last])
				last--;
			prices[2] = costs[2] + dp[last];
		}
		dp[i] = std::min(std::min(prices[0], prices[1]), prices[2]);
	}
	return dp.back();
}

std::vector<std::string> Solution::letterCombinations(std::string digits)
{
	return std::vector<std::string>();
}

std::vector<std::string> Solution::fizzBuzz(int n)
{
	std::vector<std::string> result;
	for (size_t i = 1; i <= n; i++) {
		if (0 == i % 15)
			result.emplace_back("FizzBuzz");
		else if (0 == i % 5)
			result.emplace_back("Buzz");
		else if (0 == i % 3)
			result.emplace_back("Fizz");
		else
			result.emplace_back(std::to_string(i));
	}
	return result;
}

ListNode *Solution::sortList(ListNode *head)
{
	// recursion
	if (head == nullptr || head->next == nullptr)
		return head;
	ListNode *slow = head;
	ListNode *fast = head->next;
	while (fast && fast->next) {
		slow = slow->next;
		fast = fast->next->next;
	}
	ListNode *second_half = sortList(slow->next);
	slow->next = nullptr;
	ListNode *first_half = sortList(head);
	return mergeTwoLists(first_half, second_half);
	// TODO: iteration
}

std::vector<int> Solution::countBits(int num)
{
	std::vector<int> dp(num + 1, 0);
	size_t last = 0;
	for (size_t i = 1; i <= num; i++) {
		if ((i & (i - 1)) == 0) {
			dp[i] = 1;
			last = i;
		} else {
			dp[i] = dp[last] + dp[i - last];
		}
	}
	return dp;
}

TreeNode *Solution::mergeTrees(TreeNode *t1, TreeNode *t2)
{
	if (t1 == nullptr)
		return t2;
	if (t2 == nullptr)
		return t1;
	t1->val += t2->val;
	t1->left = mergeTrees(t1->left, t2->left);
	t1->right = mergeTrees(t1->right, t2->right);
	return t1;
}

// TODO: Tree
std::vector<int>
Solution::findRightInterval(std::vector<std::vector<int> > &intervals)
{
	std::vector<int> res;
	std::map<std::pair<int, int>, int> umap;
	int idx = 0;
	for (const auto &interval : intervals)
		umap[std::make_pair(interval[0], interval[1])] = idx++;
	std::vector<std::vector<int> > sorted_intervals = intervals;
	std::sort(sorted_intervals.begin(), sorted_intervals.end());
	for (const auto &interval : intervals) {
		auto left = 0;
		auto right = intervals.size();
		if (sorted_intervals.back()[0] < interval[1])
			res.push_back(-1);
		else {
			while (left < right) {
				auto mid = left + (right - left) / 2;
				if (sorted_intervals[mid][0] - interval[1] <
				    0) // <=
					left = mid + 1;
				else
					right = mid;
			}
			res.push_back(
				umap[std::make_pair(sorted_intervals[left][0],
						    sorted_intervals[left][1])]);
		}
	}
	return res;
}

std::string Solution::intToRoman(int num)
{
#if 0
	std::string res;
	std::vector<std::vector<char>> symbols = {
		{'I', 'V'},
		{'X', 'L'},
		{'C', 'D'},
		{'M', '\0'},
	};
	int row = 0;
	while (num)
	{
		auto bit = num % 10;
		num /= 10;
		std::string substr;
		if (bit == 9) {
			substr.push_back(symbols[row + 1][0]);
			substr.push_back(symbols[row][0]);
		}
		else if (bit == 4)
		{
			substr.push_back(symbols[row][1]);
			substr.push_back(symbols[row][0]);
		}
		else
		{
			while (bit != 5 && bit != 0)
			{
				bit--;
				substr.push_back(symbols[row][0]);
			}
			if (bit == 5)
				substr.push_back(symbols[row][1]);
		}
		res.append(substr);
		row++;
	}
	std::reverse(res.begin(), res.end());
	return res;
#else
	// greedy
	std::vector<int> values = { 1000, 900, 500, 400, 100, 90, 50,
				    40,	  10,  9,   5,	 4,   1 };
	std::vector<std::string> symbols = { "M",  "CM", "D",  "CD", "C",
					     "XC", "L",	 "XL", "X",  "IX",
					     "V",  "IV", "I" };

	auto roman_digits = std::string();
	for (int i = 0; i < values.size() && 0 < num; i++)
		while (values[i] <= num) {
			num -= values[i];
			roman_digits.append(symbols[i]);
		}
	return roman_digits;
#endif // 0
}

int Solution::romanToInt(std::string s)
{
	int res = 0;
	std::map<char, int> m = { { 'I', 1 },	{ 'V', 5 },   { 'X', 10 },
				  { 'L', 50 },	{ 'C', 100 }, { 'D', 500 },
				  { 'M', 1000 } };
	for (size_t i = 0; i + 1 < s.length(); i++) {
		if (m[s[i]] < m[s[i + 1]])
			res -= m[s[i]];
		else
			res += m[s[i]];
	}
	res += m[s.back()];
	return res;
}

bool Solution::judgeCircle(std::string moves)
{
	int deltaX = 0, deltaY = 0;
	for (const auto &chr : moves) {
		switch (chr) {
		case 'R':
			deltaX++;
			break;
		case 'L':
			deltaX--;
			break;
		case 'U':
			deltaY++;
			break;
		case 'D':
			deltaY--;
			break;
		}
	}
	return deltaX == 0 && deltaY == 0;
}

int Solution::rand10()
{
	auto rand7 = []() { return rand() % 7 + 1; };
	int a = 7, b = 7;

	while (a == 7)
		a = rand7();
	while (b > 5)
		b = rand7();

	return a & 1 ? b : b + 5;
}

bool Solution::hasCycle(ListNode *head)
{
	ListNode *fast = head;
	ListNode *slow = head;
	while (fast != nullptr && fast->next != nullptr) {
		fast = fast->next->next;
		slow = slow->next;
		if (fast == slow)
			return true;
	}
	return false;
}

ListNode *Solution::getIntersectionNode(ListNode *headA, ListNode *headB)
{
	ListNode *pA = headA;
	ListNode *pB = headB;
	while (pA != pB) {
		pA = pA != nullptr ? pA->next : headB;
		pB = pB != nullptr ? pB->next : headA;
	}
	return pA;
}

std::vector<std::string>
Solution::findRestaurant(std::vector<std::string> &list1,
			 std::vector<std::string> &list2)
{
	std::vector<std::string> ret;
	std::map<std::string, size_t> map1;
	for (size_t i = 0; i < list1.size(); i++)
		map1.insert(std::make_pair(list1[i], i));
	size_t min = INT_MAX;
	for (size_t i = 0; i < list2.size(); i++) {
		if (map1.find(list2[i]) != map1.end()) {
			map1[list2[i]] += i;
			if (map1[list2[i]] < min) {
				min = map1[list2[i]];
				ret.clear();
				ret.emplace_back(list2[i]);
			} else if (map1[list2[i]] == min) {
				ret.emplace_back(list2[i]);
			}
		}
	}
	return ret;
}

//TODO:
std::string Solution::shortestParlindrome(std::string s)
{
	std::string res;
	for (int i = s.length() - 1; i >= 0; i--) {
		if (isPalindrome(s.substr(0, i))) {
			res = std::string(s.begin() + i, s.end());
			std::reverse(res.begin(), res.end());
			return res + s;
		}
	}
	res = s;
	std::reverse(res.begin(), res.end());
	return res + s;
}

std::vector<int> Solution::pancakeSort(std::vector<int> &A)
{
	// find n
	// flip 1-n.index
	auto is_sorted = [&]() {
		bool sorted = true;
		for (size_t i = 0; i < A.size(); i++)
			if (i != A[i])
				sorted = false;
		return sorted;
	};
	if (is_sorted())
		return std::vector<int>();
	std::vector<int> res;
	int idx = 0;
	while (A[idx++] != A.size())
		;
	res.push_back(idx);
	res.push_back(A.size());
	std::vector<int> subA =
		std::vector<int>(A.rbegin(), A.rbegin() + A.size() - idx);
	for (size_t i = 0; i + 1 < idx; i++)
		subA.push_back(A[i]);
	auto follows = pancakeSort(subA);
	for (const auto &num : follows)
		res.push_back(num);
	return res;
}

bool Solution::containsPattern(std::vector<int> &arr, int m, int k)
{
	return false;
}

int Solution::getMaxLen(std::vector<int> &nums)
{
	auto helper = [](std::vector<int> &nums) -> int {
		auto temp = 1;
		for (auto &&num : nums) {
			temp *= std::abs(num) / num;
		}
		if (temp == 1)
			return nums.size();
		int i = 0;
		while (0 < nums[i] && 0 < nums[nums.size() - 1 - i])
			i++;
		return -i - 1 + nums.size();
	};
	nums = { 0, 1, -2, -3, -4 };
	std::vector<std::vector<int> > withoutzero;
	std::unordered_map<int, std::vector<int> > umap;
	int start = 0;
	for (size_t i = 0; i < nums.size(); i++) {
		if (nums[i] == 0) {
			withoutzero.push_back(std::vector<int>(
				nums.begin() + start, nums.begin() + i));
			start = i + 1;
		}
	}
	withoutzero.push_back(
		std::vector<int>(nums.begin() + start, nums.end()));
	int res = 0;
	for (auto &&interval : withoutzero)
		res = std::max(helper(interval), res);
	return res;
}

// TODO: TLE
int Solution::largestComponentSize(std::vector<int> &A)
{
	//if (A.size() < 2) return A.size();
	//std::sort(A.begin(), A.end());
	//std::vector<int> temp = { A[0] };
	//std::map <int, int> gcdmap;
	//gcdmap[A[0]] = 1;
	////for (size_t i = 1; i < A.size(); i++)
	////{
	////	if (1 < std::gcd(A[0], A[i]))
	////	{
	////		temp.push_back(A[i]);
	////		gcdmap[A[i]] = 1;
	////	}
	////	for (size_t j = i + 1; j < A.size(); j++)
	////		if (1 < std::gcd(A[j], A[i]) && 0 != gcdmap.count(A[i]))
	////		{
	////			temp.push_back(A[j]);
	////			gcdmap[A[j]] = 1;
	////		}
	////}
	//int fisrt_component_size = 0;
	//while (gcdmap.size()!= fisrt_component_size)
	//{
	//	fisrt_component_size = gcdmap.size();
	//	for (const auto& e : gcdmap)
	//		for (size_t i = 0; i < A.size(); i++)
	//			if (gcdmap.count(A[i]) == 0 && 1 < std::gcd(e.first, A[i]))
	//				gcdmap[A[i]] = 1;
	//}
	//if (A.size() <= fisrt_component_size * 2)
	//	return fisrt_component_size;
	//std::vector<int> second_component;
	//for (const auto& num : A)
	//	if (gcdmap.count(num) == 0)
	//		second_component.push_back(num);
	//int second_component_size = largestComponentSize(second_component);

	//return std::max(fisrt_component_size, second_component_size);
	return 0;
}

TreeNode *removeAt(TreeNode *&root)
{
	TreeNode *curr = root;
	TreeNode *succ = nullptr;
	if (root->left == nullptr)
		root = root->right;
	else if (root->right == nullptr)
		root = root->left;
	else {
		TreeNode *target = curr;
		TreeNode *parent = nullptr;
		curr = curr->right;
		while (curr->left != nullptr) {
			parent = curr;
			curr = curr->left;
		}
		std::swap(curr->val, target->val);
		(parent == root ? parent->left : parent->right) = curr->right;
	}
	return succ;
}

TreeNode *Solution::deleteNode(TreeNode *root, int key)
{
	//{
	//	TreeNode* curr = root;
	//	TreeNode* parent = nullptr;
	//	while (curr != nullptr)
	//	{
	//		if (curr->val == key)
	//			break;
	//		if (curr->val < key)
	//			curr = curr->right;
	//		else if (key < curr->val)
	//			curr = curr->left;
	//	}

	//	TreeNode*& node= curr;
	//	if (curr->left == nullptr)
	//		node = node->right;
	//	else if (curr->right == nullptr)
	//		node = node->left;
	//	else {
	//		TreeNode* succ = curr->right;
	//		parent = curr;
	//		while (succ->left != nullptr)
	//		{
	//			parent = succ;
	//			succ = succ->left;
	//		}
	//		{
	//			curr->val ^= succ->val;
	//			succ->val ^= curr->val;
	//			curr->val ^= succ->val;
	//		}
	//		TreeNode*& node = succ;
	//		node = node->right;
	//		int kk = 0;
	//	}
	//	return root;
	//}

	TreeNode *curr = root;
	TreeNode *parent = nullptr;

	while (curr != nullptr) {
		if (curr->val == key)
			break;
		parent = curr;
		if (curr->val < key)
			curr = curr->right;
		else if (key < curr->val)
			curr = curr->left;
	}

	if (curr == nullptr)
		return root;

	if (curr->left == nullptr) {
		if (curr->right != nullptr)
			*curr = *curr->right;
		else
			curr == parent->left ? parent->left = nullptr :
						     parent->right = nullptr;
	} else if (curr->right == nullptr) {
		*curr = *curr->left;
	} else {
		TreeNode *succ = curr->right;
		parent = curr;
		while (succ->left != nullptr) {
			parent = succ;
			succ = succ->left;
		}
		std::swap(curr->val, succ->val);
		if (succ->right != nullptr)
			*succ = *succ->right;
		else
			succ == parent->left ? parent->left = nullptr :
						     parent->right = nullptr;
	}

	return root;
}

// TODO:
std::string Solution::largestTimeFromDigits(std::vector<int> &A)
{
	std::string result;
	std::vector<int> B(4, 0);
	std::sort(A.begin(), A.end());
	if (2 < A[0])
		return result;
	if (5 < A[1])
		return result;
	if (2 == A[0]) {
		if (3 < A[1])
			return result;
		if (5 < A[2])
			return result;
		if (A[2] < 4) {
			B[0] = 2;
			//B[1] ==
		}
	} else {
	}

	return result;
}

std::vector<std::vector<int> > Solution::permute(std::vector<int> &nums)
{
	if (nums.size() < 2)
		return std::vector<std::vector<int> >(1, nums);

	std::vector<std::vector<int> > result;
	for (size_t i = 0; i < nums.size(); i++) {
		std::vector<int> except_i = nums;
		except_i.erase(except_i.begin() + i);
		for (auto permutation : permute(except_i)) {
			permutation.push_back(nums[i]);
			result.emplace_back(permutation);
		}
	}
	return result;
}

bool Solution::containsNearbyAlmostDuplicate(std::vector<int> &nums, int k,
					     int t)
{
	std::map<int, std::vector<int> > map;
	for (size_t i = 0; i < nums.size(); i++)
		map[nums[i]].push_back(i);
	for (auto &&e : map)
		for (size_t i = 0; i < e.second.size(); i++)
			for (size_t minus = 0; minus <= t; minus++)
				for (size_t j = 0;
				     j < map[e.first + minus].size(); j++)
					if (e.second[i] !=
						    map[e.first + minus][j] &&
					    std::abs(e.second[i] -
						     map[e.first + minus][j]) <=
						    k)
						return true;
	return false;
}

std::vector<std::string> Solution::binaryTreePaths(TreeNode *root)
{
	const std::string arrow = "->";

	static auto dfs = make_y_combinator(
		[&](auto &&dfs, TreeNode *root, std::string str,
		    std::vector<std::string> &result) -> void {
			if (root == nullptr)
				return;
			str.append(std::to_string(root->val));
			if (root->left == nullptr && root->right == nullptr) {
				result.push_back(str);
			} else {
				str.append(arrow);
				if (root->left != nullptr) {
					dfs(root->left, str, result);
				}
				if (root->right != nullptr) {
					dfs(root->right, str, result);
				}
			}
		});

	std::vector<std::string> result;
	dfs(root, std::string(), result);
	return result;
}

std::vector<int> Solution::partitionLabels(std::string S)
{
	//S = "caedbdedda";
	std::vector<int> res;
	int left = 0;
	int right = S.length();
	std::vector<bool> flag(26, true);
	int last = 0;
	while (left < S.length()) {
		right = S.length();
		while (S[--right] != S[left])
			;
		flag[S[left] - 'a'] = false;
		while (left < right) {
			if (flag[S[left] - 'a']) {
				int end = S.length();
				while (S[--end] != S[left])
					;
				if (right < end)
					right = end;
				flag[S[left] - 'a'] = false;
			}
			left++;
		}
		left++;
		res.push_back(left - last);
		last = left;
	}
	return res;
}

std::vector<int> Solution::getAllElements(TreeNode *root1, TreeNode *root2)
{
	auto first = inorderTraversal(root1);
	auto second = inorderTraversal(root2);
	std::vector<int> res(first.size() + second.size(), 0);

	// merge
	size_t i = 0, j = 0, k = 0;
	while (j < first.size() && k < second.size()) {
		if (first[j] < second[k])
			res[i++] = first[j++];
		else
			res[i++] = second[k++];
	}
	while (j < first.size())
		res[i++] = first[j++];
	while (k < second.size())
		res[i++] = second[k++];
	return res;
}

std::string Solution::modifyString(std::string s)
{
	auto letterAfter = [](char chr) {
		return chr == 'z' ? chr = 'a' : ++chr;
	};
	for (size_t i = 0; i < s.length(); i++) {
		if (s[i] == '?') {
			char temp = 'a';
			if (i == 0) {
				while (temp == s[i + 1])
					temp = letterAfter(temp);
			} else if (i == s.length() - 1) {
				while (temp == s[i - 1])
					temp = letterAfter(temp);
			} else {
				while (temp == s[i + 1] || temp == s[i - 1])
					temp = letterAfter(temp);
			}
			s[i] = temp;
		}
	}
	return s;
}

int Solution::minCost(std::string s, std::vector<int> &cost)
{
	int res = 0;
	for (size_t i = 1; i < s.length(); i++) {
		if (s[i] == s[i - 1]) {
			if (cost[i] < cost[i - 1])
				std::swap(cost[i], cost[i - 1]);
			res += cost[i - 1];
		}
	}
	return res;
}

int Solution::numTriplets(std::vector<int> &nums1, std::vector<int> &nums2)
{
	//nums1 = { 3, 1, 2, 2 };
	//nums2 = { 1, 3, 4, 4 };
	//nums1 = { 7, 3, 4, 2, 1, 4, 1, 6, 1, 1, 5 };
	//nums2 = { 3, 5, 2, 4, 3, 1, 7, 5, 7, 5 };
	//nums1 = { 3, 5, 1, 2, 4, 3, 3, 2, 4, 2, 3, 4, 5, 2, 4, 3, 5, 3, 4, 5, 3, 1, 1, 2, 4, 2, 4, 1, 2, 1, 2, 2, 5, 2, 4, 5, 4, 5, 5, 2, 4, 4, 5, 3, 1, 2, 5, 4, 5, 1, 2 };
	//nums2 = { 3, 4, 4, 3, 3, 5, 5, 4, 3, 3, 1, 4, 5, 4, 2, 4, 2, 2, 2, 5, 4, 4, 4, 5, 2, 4, 2, 1, 2, 5, 2, 5, 5, 3, 5, 4, 3, 4, 3, 5, 1, 1, 4, 5, 3, 1, 5, 5, 5, 2, 5, 4, 2, 4, 5, 3, 2, 2 };
	auto choose2 = [](int n) { return n * (n - 1) / 2; };
	int res = 0;
	std::map<int, int> map1, map2;
	for (size_t i = 0; i < nums1.size(); i++)
		map1[nums1[i]]++;
	for (size_t i = 0; i < nums2.size(); i++)
		map2[nums2[i]]++;
	for (auto e : map1) {
		if (1 < map2[e.first])
			res += e.second * choose2(map2[e.first]);
		if (1 < map2.size()) {
			long squre = static_cast<long>(e.first) * e.first;
			auto left = map2.begin();
			auto right = map2.rbegin();
			while ((*left).first < e.first &&
			       e.first < (*right).first) {
				long product =
					static_cast<long>((*left).first) *
					(*right).first;
				if (product < squre)
					left++;
				else if (squre < product)
					right++;
				else {
					res += e.second * (*left).second *
					       (*right).second;
					left++;
					right++;
				}
			}
		}
	}
	for (auto e : map2) {
		if (1 < map1[e.first])
			res += e.second * choose2(map1[e.first]);
		if (1 < map1.size()) {
			long squre = static_cast<long>(e.first) * e.first;
			auto left = map1.begin();
			auto right = map1.rbegin();
			while ((*left).first < e.first &&
			       e.first < (*right).first) {
				long product =
					static_cast<long>((*left).first) *
					(*right).first;
				if (product < squre)
					left++;
				else if (squre < product)
					right++;
				else {
					res += e.second * (*left).second *
					       (*right).second;
					left++;
					right++;
				}
			}
		}
	}
	return res;
}

int Solution::maxNumEdgesToRemove(int n, std::vector<std::vector<int> > &edges)
{
	return 0;
}

int Solution::largestOverlap(std::vector<std::vector<int> > &A,
			     std::vector<std::vector<int> > &B)
{
	return 0;
}

std::vector<int> Solution::dailyTemperatures(std::vector<int> &T)
{
	int length = T.size();
	std::vector<int> res(length, 0);
	std::vector<int> hash(71, length);
	for (int i = T.size() - 1; i >= 0; i--) {
		int min_value = length;
		for (int temp = T[i] + 1; temp <= 100; temp++) {
			if (hash[temp - 30] == length)
				continue;
			min_value = std::min(hash[temp - 30], min_value);
			if (min_value == i + 1)
				break;
		}
		hash[T[i] - 30] = i;
		if (min_value != length)
			res[i] = min_value - i;
	}
	return res;
}

int Solution::findTargetSumWays(std::vector<int> &nums, int S)
{
	static int time = 0;
	time++;
	std::cout << time << std::endl;

	static std::map<std::pair<std::vector<int>, int>, int> dp;
	dp[std::make_pair(std::vector<int>(), 0)] = 1;
	if (nums.size() == 0)
		return S == 0 ? 1 : 0;
	int max_sum = 0;
	for (auto num : nums)
		max_sum += num;
	if (max_sum < abs(S))
		return 0;
	std::vector<int> subnums(nums.begin(), nums.begin() + nums.size() - 1);
	if (dp.find(std::make_pair(subnums, S - nums.back())) == dp.end())
		dp[std::make_pair(subnums, S - nums.back())] =
			findTargetSumWays(subnums, S - nums.back());
	if (dp.find(std::make_pair(subnums, S + nums.back())) == dp.end())
		dp[std::make_pair(subnums, S + nums.back())] =
			findTargetSumWays(subnums, S + nums.back());
	return dp[std::make_pair(subnums, S - nums.back())] +
	       dp[std::make_pair(subnums, S + nums.back())];
}

int Solution::subarraySum(std::vector<int> &nums, int k)
{
	int res = 0;
	int sum = 0;
	std::unordered_map<int, int> umap;
	for (size_t i = 0; i < nums.size(); i++) {
		umap[sum]++;
		sum += nums[i];
		if (umap.find(sum - k) != umap.end())
			res += umap[sum - k];
	}
	return res;
}

int Solution::countSubstrings(std::string s)
{
	// every single char is Palindroma
	int res = s.length();

	// Pivot is a char
	for (int i = 1; i < s.length() - 1; i++) {
		int k = 1;
		while (0 <= i - k && i + k < s.length() && s[i - k] == s[i + k])
			k++;
		res += k - 1;
	}

	// Pivot is ""
	for (int i = 1; i < s.length(); i++)
		if (s[i - 1] == s[i]) {
			res++;
			int k = 1;
			while (0 <= i - k - 1 && i + k < s.length() &&
			       s[i - k - 1] == s[i + k])
				k++;
			res += k - 1;
		}
	return res;
}

bool Solution::wordPattern(std::string pattern, std::string str)
{
	auto split = [](const std::string &str,
			const char &chr) -> std::vector<std::string> {
		std::vector<std::string> res;
		auto last = 0;
		for (size_t i = 0; i < str.length(); i++) {
			if (str[i] == chr) {
				if (last != i)
					res.emplace_back(
						std::string(str.begin() + last,
							    str.begin() + i));
				last = i + 1;
			}
		}
		if (last != str.length())
			res.emplace_back(
				std::string(str.begin() + last, str.end()));
		return res;
	};
	std::vector<std::string> word_list = split(str, ' ');
	if (word_list.size() == pattern.length()) {
		std::unordered_map<char, std::string> hash;
		std::unordered_map<std::string, char> rhash;
		for (size_t i = 0; i < word_list.size(); i++) {
			if (hash.find(pattern[i]) == hash.end() &&
			    rhash.find(word_list[i]) == rhash.end()) {
				hash[pattern[i]] = word_list[i];
				rhash[word_list[i]] = pattern[i];
			} else if (hash[pattern[i]] != word_list[i] ||
				   rhash[word_list[i]] != pattern[i])
				return false;
		}
		return true;
	}
	return false;
}

std::vector<std::vector<int> > Solution::combine(int n, int k)
{
	auto bits = [](long num) -> int {
		int count = 0;
		while (num != 0) {
			num &= num - 1;
			count++;
		}
		return count;
	};
	std::vector<std::vector<int> > res;
	for (long i = 0; i < 1 << n; i++) {
		std::vector<int> temp;
		if (bits(i) == k) {
			for (size_t bit_index = 0; bit_index < n; bit_index++) {
				if (1 == 1 & (i >> bit_index))
					temp.push_back(bit_index + 1);
			}
		}
		res.push_back(temp);
	}
	return res;
}

int Solution::sumRootToLeaf(TreeNode *root)
{
	int sum = 0;
	static auto dfs = make_y_combinator(
		[&](auto &&dfs, TreeNode *node, int num) -> void {
			if (node == nullptr)
				return;
			num += node->val;
			if (node->left == nullptr && node->right == nullptr) {
				sum += num;
				return;
			}
			dfs(node->left, num << 1);
			dfs(node->right, num << 1);
		});
	dfs(root, 0);
	return sum;
}

auto lambda = [](const std::vector<int> &candidates,
		 const std::vector<int> &nums) {
	int res = 0;
	for (size_t i = 0; i < nums.size(); i++)
		res += nums[i] * candidates[i];
	return res;
};
// TODO:
void backtrack(std::vector<int> &candidates, int target, int current_index,
	       std::vector<int> num_of_times,
	       std::vector<std::vector<int> > &result)
{
	if (candidates.size() <= current_index) //oversize()
		return;
	for (size_t i = 0;; i++) {
		num_of_times[current_index] = i;

		int sum = lambda(candidates, num_of_times);
		if (sum < target) // noconfilct
			backtrack(candidates, target, current_index + 1,
				  num_of_times, result);
		else if (sum == target) {
			std::vector<int> combination;
			for (size_t j = 0; j < num_of_times.size(); j++) {
				int time = 0;
				while (num_of_times[j] - time++ != 0)
					combination.push_back(candidates[j]);
			}
			if (!combination.empty())
				result.push_back(combination);
		} else
			return;
		num_of_times[current_index] = 0;
	}
}
std::vector<std::vector<int> >
Solution::combinationSum(std::vector<int> &candidates, int target)
{
	std::vector<std::vector<int> > result;
	std::vector<int> times(candidates.size(), 0);
	backtrack(candidates, target, 0, times, result);
	return result;
}

int Solution::numBusesToDestination(std::vector<std::vector<int> > &routes,
				    int S, int T)
{
	return 0;
}
std::vector<int> distanceKdescendants(TreeNode *root, TreeNode *except, int K);
std::vector<int> Solution::distanceK(TreeNode *root, TreeNode *target, int K)
{
	std::unordered_map<TreeNode *, TreeNode *> umap;
	auto dfs = make_y_combinator(
		[&](auto &&dfs, TreeNode *root, TreeNode *target) -> void {
			if (root == nullptr || root == target)
				return;
			if (root->left != nullptr) {
				umap[root->left] = root;
				dfs(root->left, target);
			}
			if (root->right != nullptr) {
				umap[root->right] = root;
				dfs(root->right, target);
			}
		});
	dfs(root, target);
	for (auto &&e : umap)
		std::cout << "node " << e.first->val << "'s parent is node"
			  << e.second->val << std::endl;
	std::vector<int> res;

	std::queue<TreeNode *> que;
	que.push(target);
	while (!que.empty() && 0 <= K) {
		int count = que.size();
		std::cout << K << std::endl;
		while (0 != count--) {
			TreeNode *curr = que.front();
			que.pop();
			if (K == 0)
				res.push_back(curr->val);
			if (curr->left != nullptr && umap[curr->left] != curr)
				que.push(curr->left);
			if (curr->right != nullptr && umap[curr->right] != curr)
				que.push(curr->right);
			if (umap.find(curr) != umap.end())
				que.push(umap[curr]);
		}
		K--;
	}

	return res;
}

int Solution::compareVersion(std::string version1, std::string version2)
{
	return 0;
}

int Solution::maxPathSum(TreeNode *root)
{
	if (root == nullptr)
		return 0;
	int ret = INT_MIN;
	static auto dfs = make_y_combinator([&](auto &&dfs,
						TreeNode *root) -> int {
		if (root == nullptr)
			return 0;
		int left_gain = dfs(root->left) < 0 ? 0 : dfs(root->left);
		int right_gain = dfs(root->right) < 0 ? 0 : dfs(root->right);
		ret = left_gain + root->val + right_gain;
		return std::max(left_gain + root->val, right_gain + root->val);
	});
	dfs(root);
	return ret;
}

int Solution::maxProduct(std::vector<int> &nums)
{
	int max_product = 0;
	int product[2] = { 0 };
	for (size_t i = 0; i < nums.size(); i++) {
		if (nums[i] == 0)
			product[0] = 0;
		else if (product[0] == 0)
			product[0] = nums[i];
		else
			product[0] *= nums[i];

		int j = nums.size() - i - 1;
		if (nums[j] == 0)
			product[1] = 0;
		else if (product[1] == 0)
			product[1] = nums[j];
		else
			product[1] *= nums[j];

		max_product = std::max(max_product, product[0]);
		max_product = std::max(max_product, product[1]);
	}
	return max_product;
}

std::vector<std::vector<int> > Solution::combinationSum3(int k, int n)
{
	std::vector<std::vector<int> > result;
	return result;
}

std::vector<double> Solution::averageOfLevels(TreeNode *root)
{
	std::vector<double> result;
	if (root == nullptr)
		return result;
	std::queue<TreeNode *> queue;
	queue.push(root);
	int count = 0;
	std::vector<std::vector<int> > level_order;
	while (!queue.empty()) {
		count = queue.size();
		std::vector<int> level;
		long sum_of_level = 0;
		int size = 0;
		while (0 != count--) {
			TreeNode *curr = queue.front();
			queue.pop();
			level.push_back(curr->val);
			sum_of_level += curr->val;
			size++;
			if (curr->left)
				queue.push(curr->left);
			if (curr->right)
				queue.push(curr->right);
		}
		level_order.push_back(level);
		result.push_back(static_cast<double>(sum_of_level) / size);
	}
	return result;
}

int Solution::numSpecial(std::vector<std::vector<int> > &mat)
{
	int result = 0;
	int n = mat.size();
	std::vector<int> sum_of_row(n, 0);
	std::vector<int> sum_of_colunm(n, 0);
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			sum_of_row[i] += mat[i][j];
			sum_of_colunm[i] += mat[j][i];
		}
	}
	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < n; j++)
			if (mat[i][j] == 1 && sum_of_row[i] == 1 &&
			    sum_of_colunm[j] == 1)
				result++;
	return result;
}

int Solution::unhappyFriends(int n, std::vector<std::vector<int> > &preferences,
			     std::vector<std::vector<int> > &pairs)
{
	int res = 0;
	std::vector<bool> unhappy(n, false);
	std::map<int, int> pair_map;
	for (auto pair : pairs) {
		pair_map[pair[0]] = pair[1];
		pair_map[pair[1]] = pair[0];
	}
	for (int i = 0; i < n; i++) {
		if (unhappy[i] == false) {
			for (size_t j = 0; j < preferences[i].size(); j++) {
				if (preferences[i][j] == pair_map[i])
					break;
				//check preferences[pair[0]][i] happy
				for (size_t k = 0; k < preferences[j].size();
				     k++) {
					if (preferences[preferences[i][j]][k] ==
					    pair_map[preferences[i][j]])
						break;
					if (preferences[preferences[i][j]][k] ==
					    i) {
						unhappy[i] = true;
						unhappy[preferences[i][j]] =
							true;
					}
				}
			}
		}
	}
	for (size_t i = 0; i < unhappy.size(); i++)
		if (unhappy[i])
			res++;
	return res;
}

// TODO:
int Solution::minCostConnectPoints(std::vector<std::vector<int> > &points)
{
	static auto manhattan_distance = [](std::vector<int> &pointA,
					    std::vector<int> &pointB) {
		return std::abs(pointA[0] - pointB[0]) +
		       std::abs(pointA[1] - pointB[1]);
	};
	int res = 0;
	struct edge {
		int v;
		int w;
		int weight;
		edge(int _v, int _w, int _weight)
			: v(_v), w(_w), weight(_weight)
		{
		}
		bool operator<(edge other)
		{
			return weight < other.weight;
		}
		bool operator>(edge other)
		{
			return weight > other.weight;
		}
	};
#if false
	// Kruskal's / Union Find
	std::priority_queue<edge, std::vector<edge>, std::greater<>> pq;
	for (size_t i = 0; i + 1 < points.size(); i++)
	{
		for (size_t j = i + 1; j < points.size(); j++)
		{
			pq.push(edge(i, j, manhattan_distance(points[i], points[j])));
		}
	}

	//quick union
	std::vector<int> id(points.size(), 0);
	std::vector<int> sz(points.size(), 0);
	for (size_t i = 0; i < points.size(); i++)
		id[i] = i;
	auto root = [&](int i) -> int {
		while (id[i] != i) {
			id[i] = id[id[i]];
			i = id[i];
		}
		return i;
	};
	auto is_connected = [&](int p, int q) -> bool {
		return root(p) == root(q);
	};
	auto connect = [&](int p, int q) -> void {
		int i = root(p);
		int j = root(q);
		if (sz[i] < sz[j]) {
			id[i] = j;
			sz[j] += i;
		}
		else {
			id[j] = i;
			sz[i] += j;
		}
	};
	int edge_used = 0;
	while (/*!pq.empty() || */edge_used < points.size() - 1)
	{
		auto e = pq.top();
		pq.pop();
		if (!is_connected(e.v, e.w))
		{
			connect(e.v, e.w);
			res += e.weight;
			edge_used++;
		}
	}
#elif false
	// Prim's algorithm Lazy version
	std::priority_queue<edge, std::vector<edge>, std::greater<> > pq;
	std::vector<bool> marked(points.size(), false);
	int edge_used = 0;
	marked[0] = true;
	for (size_t i = 1; i < points.size(); i++)
		pq.push(edge(0, i, manhattan_distance(points[0], points[i])));
	while (!pq.empty() && edge_used + 1 < points.size()) {
		while (marked[pq.top().v] && marked[pq.top().w])
			pq.pop();
		auto &e = pq.top();
		res += e.weight;
		int current = marked[e.v] ? e.w : e.v;
		pq.pop();

		// visit current
		marked[current] = true;
		edge_used++;
		for (size_t i = 0; i < points.size(); i++)
			if (!marked[i])
				pq.push(edge(current, i,
					     manhattan_distance(points[current],
								points[i])));
	}
#else // Prim's algorithm Lazy version
	std::vector<bool> marked(points.size(), false);
	std::vector<int> dist_to(points.size(), 0.0);
	int edge_used = 0;
	marked[0] = true;
	for (size_t i = 1; i < points.size(); i++) {
		dist_to[i] = manhattan_distance(points[0], points[i]);
	}
	while (edge_used + 1 < points.size()) {
		int current = 0;
		int distance = INT_MAX;

		// TODO: make dist_to a map?;
		for (size_t i = 0; i < dist_to.size(); i++) {
			if (marked[i])
				continue;
			if (dist_to[i] < distance) {
				current = i;
				distance = dist_to[i];
			}
		}

		marked[current] = true;
		res += distance;
		edge_used++;

		for (size_t i = 0; i < points.size(); i++)
			if (!marked[i])
				dist_to[i] = std::min(
					dist_to[i],
					manhattan_distance(points[current],
							   points[i]));
	}
#endif
	return res;
}

// TODO:
bool Solution::isTransformable(std::string s, std::string t)
{
	std::string sorted_s = s;
	std::string sorted_t = t;
	std::sort(sorted_s.begin(), sorted_s.end());
	std::sort(sorted_t.begin(), sorted_t.end());
	if (sorted_s != sorted_t)
		return false;

	int count[2] = { 0 };
	for (size_t i = 0; i + 1 < s.length(); i++) {
		for (size_t j = i + 1; j < s.length(); j++) {
			if (s[j] < s[i])
				count[0]++;
			if (t[j] < t[i])
				count[1]++;
		}
	}
	return count[1] < count[0];
}

std::vector<std::vector<int> >
Solution::insert(std::vector<std::vector<int> > &intervals,
		 std::vector<int> &newInterval)
{
	std::vector<std::vector<int> > result;
	for (auto interval : intervals) {
		if (interval[1] < newInterval[0] ||
		    newInterval[1] < interval[0])
			result.push_back(interval);
		else {
			newInterval[0] = std::min(newInterval[0], interval[0]);
			newInterval[1] = std::max(newInterval[1], interval[1]);
		}
	}
	auto it = result.begin();
	for (;; it++) {
		if (newInterval[0] < (*it)[0])
			break;
	}
	result.insert(it, newInterval);
	return result;
}

int Solution::lengthOfLastWord(std::string s)
{
	int length = 0;
	int last = s.length();
	for (int i = s.length() - 1; 0 <= i; i--) {
		if (s[i] == ' ') {
			length = last - 1 - i;
			last = i;
		}
		if (length != 0)
			return length;
	}
	return last;
}

int Solution::getSum(int a, int b)
{
	int res = 0;
	int carry = 0;
	for (int i = 0; i < 32; i++) {
		if ((a >> i) & 1 && (b >> i) & 1) {
			res |= carry << i;
			carry = 1;
		} else if ((a >> i) & 1 || (b >> i) & 1) {
			if (carry == 0)
				res |= 1 << i;
		} else {
			res |= carry << i;
			carry = 0;
		}
	}
	return res;
}

int Solution::strStr(std::string haystack, std::string needle)
{
	// brute force
	for (size_t i = 0; i < haystack.size() - needle.size() + 1; i++) {
		bool flag = true;
		for (size_t j = 0; j < needle.size(); j++) {
			if (haystack[i + j] != needle[j]) {
				flag = false;
				break;
			}
		}
		if (flag)
			return i;
	}
	return -1;
	// KMP

	// BM

	return 0;
}

// TODO: delete
int Solution::findMaximumXOR(std::vector<int> &nums)
{
	struct TrieNode {
		std::map<int, TrieNode *> children;
		bool is_end = false;
	};
	// findbits
	int max_num = 0;
	for (auto num : nums) {
		if (max_num < num)
			max_num = num;
	}
	int bit_count = 0;
	while (max_num >> bit_count != 0)
		bit_count++;

	//insert to Trie
	TrieNode *root = new TrieNode();
	TrieNode *curr = root;
	for (auto num : nums) {
		//insert
		curr = root;
		for (int i = bit_count - 1; i >= 0; i--) {
			auto bit = (num >> i) & 1;
			if (curr->children.find(bit) == curr->children.end()) {
				TrieNode *node = new TrieNode();
				curr->children[bit] = node;
			}
			curr = curr->children[bit];
		}
		curr->is_end = true;
	}

	int res = 0;
	for (auto num : nums) {
		if ((1 << (bit_count - 1)) <= num) {
			curr = root;
			int toXor = 0;
			for (int i = bit_count - 1; i >= 0; i--) {
				auto bit = (num >> i) & 1;
				toXor <<= 1;
				if (curr->children.find(1 - bit) !=
				    curr->children.end()) {
					curr = curr->children[1 - bit];
					toXor += 1 - bit;
				} else {
					curr = curr->children[bit];
					toXor += bit;
				}
			}
			res = std::max(res, num ^ toXor);
		}
	}
	return res;
}

bool Solution::isRobotBounded(std::string instructions)
{
	const double pi = std::acos(-1);
	struct coord {
		int x = 0;
		int y = 0;
		void operator+=(coord other)
		{
			x += other.x;
			y += other.y;
		}
		bool operator==(coord other)
		{
			return x == other.x && y == other.y;
		}
		void rotate(double rad)
		{
			auto temp = std::cos(rad) * x - std::sin(rad) * y;
			y = std::sin(rad) * x + std::cos(rad) * y;
			x = temp;
		}
	};
	coord position = { 0, 0 };
	coord direction = { 0, 1 };
	for (auto letter : instructions) {
		if (letter == 'G') {
			//move_along_direction;
			position += direction;
		} else {
			//change direction
			if (letter == 'L')
				direction.rotate(-pi / 2.0);
			else
				direction.rotate(pi / 2.0);
		}
	}

	return direction.x != 0 || direction.y != 1 ||
	       (position.x == 0 && position.y == 0);
}

int Solution::countPrimes(int n)
{
	int count = 0;
	std::vector<int> nums(n, 0);
	std::vector<bool> marks(n, true);
	for (size_t i = 1; i < nums.size(); i++)
		nums[i] = i;

	for (size_t i = 2; i < nums.size(); i++) {
		if (marks[i] == false)
			continue;
		for (size_t j = 2; j < nums.size(); j++) {
			if (n <= i * j)
				break;
			marks[i * j] = false;
		}
	}
	for (size_t i = 1; i < marks.size(); i++)
		if (marks[i])
			count++;
	return count;
}

std::string Solution::simplifyPath(std::string path)
{
	static auto split = [](const std::string &str,
			       const char &chr) -> std::vector<std::string> {
		std::vector<std::string> res;
		auto last = 0;
		for (size_t i = 0; i < str.length(); i++) {
			if (str[i] == chr) {
				if (last != i)
					res.emplace_back(
						std::string(str.begin() + last,
							    str.begin() + i));
				last = i + 1;
			}
		}
		if (last != str.length())
			res.emplace_back(
				std::string(str.begin() + last, str.end()));
		return res;
	};
	std::vector<std::string> directorys = split(path, '/');
	std::vector<std::string> s;
	std::string result;
	for (auto r : directorys) {
		if (r == "..") {
			if (!s.empty())
				s.resize(s.size() - 1);
		} else if (r == ".")
			;
		else {
			s.push_back(r);
		}
	}
	for (auto r : s)
		result.append("/" + r);
	return result;
}

void dfs(TreeNode *root, std::vector<std::vector<std::string> > &grid, int row,
	 int column)
{
	if (nullptr == root)
		return;
	grid[row][column] = std::to_string(root->val);
	row++;
	if (row < grid.size()) {
		std::cout << grid.size() << std::endl;
		int height = grid.size() - row - 1;
		std::cout << height << std::endl;
		int delta = 1 << height;
		std::cout << delta << std::endl;
		dfs(root->left, grid, row, column - delta);
		dfs(root->right, grid, row, column + delta);
	}
}
std::vector<std::vector<std::string> > Solution::printTree(TreeNode *root)
{
	static auto height = make_y_combinator([](auto &&height,
						  TreeNode *node) -> int {
		if (node == nullptr)
			return -1;
		return std::max(height(node->left), height(node->right)) + 1;
	});
	static auto print_helper = make_y_combinator(
		[](auto &&print_helper, TreeNode *root,
		   std::vector<std::vector<std::string> > &grid, int row,
		   int column, int delta_column) -> void {
			if (nullptr == root)
				return;
			grid[row][column] = std::to_string(root->val);
			row++;
			delta_column >>= 1;
			print_helper(root->left, grid, row,
				     column - delta_column, delta_column);
			print_helper(root->right, grid, row,
				     column + delta_column, delta_column);
		});
	int height_of_root = height(root);
	std::vector<std::vector<std::string> > result(
		height_of_root + 1,
		std::vector<std::string>((1 << (height_of_root + 1)) - 1, ""));
	print_helper(root, result, 0, (1 << height_of_root) - 1,
		     1 << height_of_root);
	return result;
}

bool Solution::isCompleteTree(TreeNode *root)
{
	return false;
}

std::vector<int> Solution::sequentialDigits(int low, int high)
{
	// enum
	std::vector<int> ans;
	for (int i = 1; i <= 9; ++i) {
		int num = i;
		for (int j = i + 1; j <= 9; ++j) {
			num = num * 10 + j;
			if (num >= low && num <= high) {
				ans.push_back(num);
			}
		}
	}
	std::sort(ans.begin(), ans.end());
	return ans;
}

std::string Solution::reorderSpaces(std::string text)
{
	auto split = [](const std::string &str,
			const char &chr) -> std::vector<std::string> {
		std::vector<std::string> res;
		auto last = 0;
		for (size_t i = 0; i < str.length(); i++) {
			if (str[i] == chr) {
				if (last != i)
					res.emplace_back(
						std::string(str.begin() + last,
							    str.begin() + i));
				last = i + 1;
			}
		}
		if (last != str.length())
			res.emplace_back(
				std::string(str.begin() + last, str.end()));
		return res;
	};
	std::string res;
	std::vector<std::string> words = split(text, ' ');
	int len = text.length();
	for (auto &&word : words)
		len -= word.length();
	int per = len;
	if (1 < words.size())
		per /= words.size() - 1;
	for (auto &&word : words) {
		res.append(word);
		for (size_t i = 0; i < per; i++)
			res.push_back(' ');
	}
	res.resize(len);
	return res;
}

// TODO: backtrack + dfs(append last / next);
int Solution::maxUniqueSplit(std::string s)
{
	std::map<std::string, bool> apmap;
	std::vector<std::string> strs;
	int count = 0;
	for (size_t i = 0; i < s.length();) {
		for (size_t j = 1; i + j <= s.length(); j++) {
			auto str = s.substr(i, j);
			if (apmap.end() == apmap.find(str) ||
			    false == apmap[str]) {
				apmap[str] = true;
				strs.emplace_back(str);
				i += j;
				count++;
				break;
			} else {
				apmap[strs.back()] = false;
				strs.back().append(str);
				apmap[strs.back()] = true;
				i += j;
				break;
			}
		}
	}
	for (auto str : strs) {
		std::cout << str << std::endl;
	}
	return count;
}
long max_value = -1;
void dfs(std::vector<std::vector<int> > &grid, long value, int row, int column,
	 int row_size, int column_size)
{
	value *= grid[row][column];
	if (0 == value) {
		max_value = std::max(value, max_value);
		return;
	}
	if (row < row_size - 1)
		dfs(grid, value, row + 1, column, row_size, column_size);
	if (column < column_size - 1)
		dfs(grid, value, row, column + 1, row_size, column_size);
	if (column_size - 1 == column && row_size - 1 == row)
		max_value = std::max(value, max_value);
}
int Solution::maxProductPath(std::vector<std::vector<int> > &grid)
{
	max_value = -1;
	dfs(grid, 1, 0, 0, grid.size(), grid[0].size());
	if (1000000007 < max_value)
		max_value %= 1000000007;
	return max_value;
}

TreeNode *Solution::convertBST(TreeNode *root)
{
	TreeNode *predecessor = nullptr;
	TreeNode *curr = root;
	int sum = 0;
	auto visit = [&](TreeNode *&node) {
		node->val += sum;
		sum = node->val;
	};
	while (curr) {
		if (!curr->right) {
			visit(curr);
			std::cout << curr->val << std::endl;
			curr = curr->left;
		} else {
			predecessor = curr->right;
			while (predecessor->left && predecessor->left != curr)
				predecessor = predecessor->left;
			if (predecessor->left == curr) {
				predecessor->left = nullptr;
				visit(curr);
				curr = curr->left;
			} else {
				predecessor->left = curr;
				curr = curr->right;
			}
		}
	}
	return root;
}

bool Solution::carPooling(std::vector<std::vector<int> > &trips, int capacity)
{
	enum { num_passengers, start_location, end_location };
	std::map<int, int> passengers_changed;
	for (const auto &trip : trips) {
		passengers_changed[trip[start_location]] +=
			trip[num_passengers];
		passengers_changed[trip[end_location]] -= trip[num_passengers];
	}
	int current_num_passengers = 0;
	for (const auto &e : passengers_changed) {
		current_num_passengers += e.second;
		if (capacity < current_num_passengers)
			return false;
	}
	return true;
}

int Solution::minCameraCover(TreeNode *root)
{
	return 0;
}

int Solution::distributeCoins(TreeNode *root)
{
	return 0;
}

void Solution::nextPermutation(std::vector<int> &nums)
{
	//for (int i = nums.size() - 1; i - 1>= 0; i--)
	//	if (nums[i - 1] < nums[i])
	//		for (int j = nums.size() - 1; j >= i; j--)
	//			if (nums[i - 1] < nums[j]) {
	//				std::swap(nums[i - 1], nums[j]);
	//				std::reverse(nums.begin() + i, nums.end());
	//				return;
	//			}
	//std::reverse(nums.begin(), nums.end());
	auto _UFirst = nums.begin();
	const auto _ULast = nums.end();
	auto _UNext = _ULast;
	if (_UFirst == _ULast || _UFirst == --_UNext) {
		return; // return false;
	}
	for (;;) { // find rightmost element smaller than successor
		auto _UNext1 = _UNext;
		if (*--_UNext <
		    *_UNext1) { // swap with rightmost element that's smaller, flip suffix
			auto _UMid = _ULast;
			do {
				--_UMid;
			} while (!(*_UNext < *_UMid));

			std::iter_swap(_UNext, _UMid);
			std::reverse(_UNext1, _ULast);
			return; // return true;
		}

		if (_UNext == _UFirst) { // pure descending, flip all
			std::reverse(_UFirst, _ULast);
			return; // return false;
		}
	}
}

int Solution::uniquePathsWithObstacles(
	std::vector<std::vector<int> > &obstacleGrid)
{
	if (obstacleGrid.empty() || obstacleGrid[0].empty() ||
	    1 == obstacleGrid.back().back())
		return 0;
	std::vector<std::vector<int> > dp(
		obstacleGrid.size(),
		std::vector<int>(obstacleGrid[0].size(), 0));

	dp.back().back() = 1;
	for (int row = obstacleGrid.size() - 2; row >= 0; row--) {
		if (1 == obstacleGrid[row][obstacleGrid[0].size() - 1])
			continue;
		dp[row][obstacleGrid[0].size() - 1] =
			dp[row + 1][obstacleGrid[0].size() - 1];
	}
	for (int column = obstacleGrid.size() - 2; column >= 0; column--) {
		if (1 == obstacleGrid[obstacleGrid.size() - 1][column])
			continue;
		dp[obstacleGrid.size() - 1][column] =
			dp[obstacleGrid.size() - 1][column + 1];
	}
	for (int row = obstacleGrid.size() - 2; row >= 0; row--) {
		for (int column = obstacleGrid[0].size() - 2; column >= 0;
		     column--) {
			if (1 == obstacleGrid[row][column])
				continue;
			dp[row][column] =
				dp[row + 1][column] + dp[row][column + 1];
		}
	}
	return dp[0][0];
}

int Solution::uniquePathsIII(std::vector<std::vector<int> > &grid)
{
	return 0;
}

std::vector<int> Solution::majorityElement(std::vector<int> &nums)
{
	std::deque<int> dq;
	LONG_MAX;
	int count[2] = { 0 };
	int candidate[2] = { 0 };
	for (auto num : nums) {
		if (candidate[0] == num)
			count[0]++;
		else if (candidate[1] == num)
			count[1]++;
		else if (count[0] == 0) {
			count[0] = 1;
			candidate[0] = num;
		} else if (count[1] == 0) {
			count[1] = 1;
			candidate[1] = num;
		} else {
			count[0]--;
			count[1]--;
		}
	}
	count[0] = 0;
	count[1] = 0;
	for (auto num : nums) {
		if (candidate[0] == num)
			count[0]++;
		else if (candidate[1] == num)
			count[1]++;
	}
	std::vector<int> result;
	if (nums.size() / 3 < count[0])
		result.push_back(candidate[0]);
	if (nums.size() / 3 < count[1])
		result.push_back(candidate[1]);
	return result;
}

int Solution::connectTwoGroups(std::vector<std::vector<int> > &cost)
{
	return 0;
}

int partition_(std::vector<int> &nums, size_t first, size_t last)
{
#if true
	std::swap(nums[first], nums[first + rand() % (last - first)]);
	const auto pivot = nums[first];
	size_t left = first, right = last - 1;
	while (left < right) {
		while (left < right && pivot < nums[right])
			right--;
		if (left < right)
			nums[left++] = nums[right];
		while (left < right && nums[left] < pivot)
			left++;
		if (left < right)
			nums[right--] = nums[left];
	}
	assert(left == right);
	nums[left] = pivot;
	return left;
#else
	std::swap(nums[first], nums[first + rand() % (last - first)]);
	const auto pivot = nums[first];
	size_t mid = first;
	for (size_t k = first + 1; k < last; k++) {
		if (nums[k] < pivot)
			std::swap(nums[++mid], nums[k]);
	}
	std::swap(nums[first], nums[mid]);
	return mid;
#endif
}
int Solution::findKthLargest(std::vector<int> &nums, int k)
{
	int first = 0, last = nums.size();
	int index = nums.size() - k;
	while (first < last) {
		int pivot_index = partition_(nums, first, last);
		if (pivot_index < index)
			first = pivot_index + 1;
		else
			last = pivot_index;
	}
	return nums[first];
}

void recQuickSort(std::vector<int> &nums, size_t first, size_t last)
{
	if (first < last) {
		size_t pivot_location = partition_(nums, first, last);
		recQuickSort(nums, first, pivot_location);
		recQuickSort(nums, pivot_location + 1, last);
	}
}
std::vector<int> Solution::sortArray(std::vector<int> &nums)
{
	recQuickSort(nums, 0, nums.size());
	return nums;
}

int Solution::canCompleteCircuit(std::vector<int> &gas, std::vector<int> &cost)
{
	int N = gas.size();
	int sum = 0;
	int minimum = INT_MAX;
	int index_of_minimum = -1;
	for (size_t i = 0; i < N; i++) {
		sum += gas[i] - cost[i];
		if (sum < minimum) {
			minimum = sum;
			index_of_minimum = i;
		}
	}
	return sum < 0 ? -1 : (index_of_minimum + 1) % N;
}

std::vector<int> Solution::findMode(TreeNode *root)
{
	std::vector<int> res;
	int count = 0;
	int max_count = 0;
	int current_num = 0;
	auto visit = [&](const TreeNode *node) {
		if (node->val == current_num) {
			count++;
		} else {
			count = 1;
			current_num = node->val;
		}
		if (max_count < count) {
			max_count = count;
			res.clear();
			res.push_back(current_num);
		} else if (max_count == count) {
			res.push_back(current_num);
		}
	};

	TreeNode *pred = nullptr;
	TreeNode *curr = root;
	while (curr) {
		if (curr->left == nullptr) {
			visit(curr);
			curr = curr->right;
		} else {
			pred = curr->left;
			while (pred->right != nullptr && pred->right != curr)
				pred = pred->right;
			if (pred->right == nullptr) {
				pred->right = curr;
				curr = curr->left;
			} else {
				pred->right = nullptr;
				visit(curr);
				curr = curr->right;
			}
		}
	}
	return res;
}

// TODO:
std::vector<int> Solution::postorderTraversal(TreeNode *root)
{
	std::vector<int> res;
	auto visit = [&](const TreeNode *node) { res.push_back(node->val); };
	TreeNode *pred = nullptr;
	TreeNode *curr = root;
	TreeNode dummy;
	curr = &dummy;
	curr->left = root;
	while (curr) {
		if (curr->left == nullptr && curr->right == nullptr) {
			visit(curr);
			curr = curr->right; //parent
		} else {
			if (curr->right != nullptr)
				pred = curr->right;
			else
				pred = curr->left;
		}
	}
	return res;
	return std::vector<int>();
}

char Solution::findTheDifference(std::string s, std::string t)
{
	std::unordered_map<char, bool> umap;
	for (auto letter : t) {
		umap[letter] = !umap[letter];
	}
	for (auto letter : s) {
		umap[letter] = !umap[letter];
	}
	for (auto e : umap) {
		if (e.second == true)
			return e.first;
	}
	return 0;
}

std::string Solution::largestNumber(std::vector<int> &nums)
{
	std::vector<std::string> num_strs;
	for (auto num : nums) {
		num_strs.emplace_back(std::to_string(num));
	}
	std::sort(num_strs.begin(), num_strs.end(),
		  [](const auto &lhs, const auto &rhs) {
			  return lhs + rhs > rhs + lhs;
		  });
	std::string ret;
	for (const auto &num : num_strs)
		ret.append(num);
	return ret;
}

int Solution::findPoisonedDuration(std::vector<int> &timeSeries, int duration)
{
	long result = timeSeries.size() * duration;
	for (size_t i = 0; i < timeSeries.size() - 1; i++) {
		if (timeSeries[i + 1] - timeSeries[i] < duration)
			result -=
				duration - (timeSeries[i + 1] - timeSeries[i]);
	}
	return static_cast<int>(result);
}

// TODO:
TreeNode *Solution::lowestCommonAncestor(TreeNode *root, TreeNode *p,
					 TreeNode *q)
{
	auto dfs = make_y_combinator([&](auto &&dfs, TreeNode *node) {});
	return nullptr;
}

int Solution::minOperations(std::vector<std::string> &logs)
{
	int res = 0;
	for (size_t i = 0; i < logs.size(); i++) {
		if (logs[i] == "./") {
			;
		} else if (logs[i] == "../") {
			res--;
		} else {
			res++;
		}
		if (res < 0)
			res = 0;
	}
	return res;
}

int Solution::minOperationsMaxProfit(std::vector<int> &customers,
				     int boardingCost, int runningCost)
{
	if (boardingCost * 4 <= runningCost)
		return -1;
	std::vector<int> profits;
	int total = 0;
	int board = 0;
	int profited = 0;
	for (size_t i = 0; i < customers.size(); i++) {
		total += customers[i];
		if (total < 4)
			board = total;
		else
			board = 4;
		total -= board;
		if (0 < board) {
			profits.push_back(profited + board * boardingCost -
					  runningCost);
			profited = profits.back();
		}
	}
	while (0 < total) {
		if (total < 4)
			board = total;
		else
			board = 4;
		total -= board;
		if (0 < board) {
			profits.push_back(profited + board * boardingCost -
					  runningCost);
			profited = profits.back();
		}
	}
	int max_profit = 0;
	int index = 0;
	for (size_t i = 0; i < profits.size(); i++) {
		if (max_profit < profits[i]) {
			index = i;
			max_profit = profits[i];
		}
	}
	return index + 1;
}

int Solution::maximumRequests(int n, std::vector<std::vector<int> > &requests)
{
	std::vector<int> buildings(n, 0);
	for (auto request : requests) {
		buildings[request[0]]--;
		buildings[request[1]]++;
	}
	int result = 0;
	for (auto build : buildings) {
		if (build != 0) {
			result += abs(build) / 2 + build % 2;
		}
	}
	return requests.size() - result;
}

std::vector<double>
Solution::calcEquation(std::vector<std::vector<std::string> > &equations,
		       std::vector<double> &values,
		       std::vector<std::vector<std::string> > &queries)
{
	return std::vector<double>();
}

int Solution::findMinArrowShots(std::vector<std::vector<int> > &points)
{
	std::sort(points.begin(), points.end());
	int count = 0;
	int min_end = INT_MIN;
	int result = 0;
	for (const auto &point : points) {
		if (count == 0) {
			count = 1;
			min_end = point[1];
			result++;
		} else {
			if (point[0] <= min_end) {
				count++;
				min_end = std::min(point[1], min_end);
			} else {
				count = 1;
				min_end = point[1];
				result++;
			}
		}
	}
	return result;
}

std::vector<int>
Solution::findRedundantConnection(std::vector<std::vector<int> > &edges)
{
	std::vector<int> root(edges.size() + 1);
	for (size_t i = 0; i < root.size(); i++) {
		root[i] = i;
	}
	size_t i = 0;
	auto find = [&](int p) -> int {
		while (p != root[p])
			p = root[p];
		return p;
	};
	for (; i < edges.size(); i++) {
		if (find(edges[i][0]) == find(edges[i][1]))
			break;
		root[find(edges[i][1])] = find(edges[i][0]);
	}
	return edges[i];
}

// TODO:
std::vector<int>
Solution::findRedundantDirectedConnection(std::vector<std::vector<int> > &edges)
{
	std::vector<int> root(edges.size() + 1);
	for (size_t i = 0; i < root.size(); i++) {
		root[i] = i;
	}
	size_t i = 0;
	auto find = [&](int p) -> int {
		while (p != root[p])
			p = root[p];
		return p;
	};
	for (; i < edges.size(); i++) {
		if (find(edges[i][0]) == find(edges[i][1]))
			break;
		root[find(edges[i][1])] = find(edges[i][0]);
	}
	return edges[i];
}

std::string Solution::removeDuplicateLetters(std::string s)
{
	std::deque<char> dq;
	return std::string();
}

bool Solution::buddyStrings(std::string A, std::string B)
{
	if (A.length() < 2 || A.length() != B.length())
		return false;
	if (A == B) {
		std::set<char> set(A.begin(), A.end());
		return set.size() < A.length();
	}
	size_t left = 0, right = A.length() - 1;
	while (true) {
		if (left + 1 == right)
			break;
		if (A[left] == B[left]) {
			left++;
			continue;
		} else if (A[right] == B[right]) {
			right--;
			continue;
		} else {
			break;
		}
	}
	std::swap(A[left], A[right]);
	return A == B;
}

std::vector<std::string> Solution::commonChars(std::vector<std::string> &A)
{
	if (A.empty())
		return std::vector<std::string>();
	std::vector<std::vector<int> > times(A.size(), std::vector<int>(26, 0));
	std::vector<std::string> result;
	for (size_t i = 0; i < A.size(); i++) {
		for (auto letter : A[i])
			times[i][letter - 'a']++;
	}
	for (size_t i = 0; i < 26; i++) {
		int common_times = INT_MAX;
		for (size_t j = 0; j < A.size(); j++) {
			if (times[j][i] < common_times)
				common_times = times[j][i];
		}
		while (0 < common_times) {
			result.emplace_back(std::string(1, i + 'a'));
			common_times--;
		}
	}
	return result;
}

bool Solution::searchMatrix(std::vector<std::vector<int> > &matrix, int target)
{
	if (matrix.empty() || matrix[0].empty())
		return false;

	// find the last row matrix[row][0] <= target
	size_t top = 0;
	size_t bottom = matrix.size() - 1;
	while (top < bottom) {
		auto row = top + (bottom - top + 1) / 2; // last
		if (!(matrix[row][0] <= target))
			bottom = row - 1;
		else
			top = row;
	}

	// binary search in the row
	// find the first e[0] >= target
	size_t left = 0, right = matrix[0].size() - 1;
	while (left < right) {
		auto column = left + (right - left) / 2;
		if (matrix[top][column] < target)
			left = column + 1;
		else // !(e[0] >= target)
			right = column;
	}

	return matrix[top][left] == target;
}

std::vector<int> Solution::asteroidCollision(std::vector<int> &asteroids)
{
	std::vector<int> stk;
	for (auto asteroid : asteroids) {
		if (stk.empty()) {
			stk.push_back(asteroid);
		} else if (stk.back() < 0) {
			stk.push_back(asteroid);
		} else if (asteroid < 0) {
			// collision
			while (!stk.empty() && 0 < stk.back() &&
			       asteroid + stk.back() < 0)
				stk.pop_back();
			if (stk.empty() || stk.back() < 0)
				stk.push_back(asteroid);
			else if (asteroid + stk.back() == 0)
				stk.pop_back();
		} else {
			stk.push_back(asteroid);
		}
	}
	return stk;
}

bool Solution::find132pattern(std::vector<int> &nums)
{
	if (nums.size() < 3)
		return false;
	std::vector<int> min(nums.size());
	int min_value = nums[0];
	for (size_t i = 0; i < nums.size() - 1; i++) {
		if (nums[i] < min_value)
			min_value = nums[i];
		min[i + 1] = min_value;
	}

	std::stack<int> stk;
	stk.push(nums.back());
	for (int j = nums.size() - 2; j >= 1; j--) {
		if (min[j] < nums[j]) {
			while (!stk.empty() && stk.top() <= min[j]) {
				stk.pop();
			}
			if (!stk.empty() && stk.top() < nums[j]) {
				return true;
			}
			// stk.empty() || stk.top() >= nums[j]
			stk.push(nums[j]);
		}
	}
	return false;
}

bool Solution::winnerSquareGame(int n)
{
	std::vector<bool> dp(n + 1);
	dp[0] = false;
	dp[1] = true;
	for (size_t i = 2; i < n + 1; i++) {
		auto sqt = static_cast<int>(std::sqrt(i));
		if (sqt * sqt == i) {
			dp[i] = true;
			continue;
		}

		for (size_t j = 1; j <= sqt; j++) {
			if (!dp[i - j * j]) {
				dp[i] = true;
				break;
			}
		}
	}
	return dp.back();
}

double Solution::champagneTower(int poured, int query_row, int query_glass)
{
	size_t height = query_row + 2;
	std::vector<std::vector<float> > tower(
		height, std::vector<float>(height, 0.0f));
	tower[0][0] = static_cast<float>(poured);
	float tmp = 0.0f;
	for (size_t i = 0; i <= query_row; i++) {
		for (size_t j = 0; j <= i; j++) {
			if (1.0f < tower[i][j]) {
				tmp = tower[i][j] - 1.0f;
				tower[i][j] = 1.0f;
				tower[i + 1][j] += tmp / 2.0f;
				tower[i + 1][j + 1] += tmp / 2.0f;
			}
		}
	}
	return static_cast<double>(tower[query_row][query_glass]);
}

int Solution::longestConsecutive(std::vector<int> &nums)
{
	std::unordered_map<int, int> hashmap;
	int ret = 0;
	for (auto &&num : nums) {
		if (hashmap.find(num) != hashmap.end())
			continue;
		hashmap[num] = 1;
		if (INT_MIN < num && hashmap.find(num - 1) != hashmap.end())
			hashmap[num] += hashmap[num - 1];
		if (num < INT_MAX && hashmap.find(num + 1) != hashmap.end())
			hashmap[num] += hashmap[num + 1];
		// update
		if (INT_MIN < num && hashmap.find(num - 1) != hashmap.end())
			hashmap[num - hashmap[num - 1]] = hashmap[num];
		if (num < INT_MAX && hashmap.find(num + 1) != hashmap.end())
			hashmap[num + hashmap[num + 1]] = hashmap[num];
		ret = std::max(ret, hashmap[num]);
	}
	return ret;
}

ListNode *Solution::detectCycle(ListNode *head)
{
	ListNode *fast = head;
	ListNode *slow = head;
	while (fast != nullptr && fast->next != nullptr) {
		fast = fast->next->next;
		slow = slow->next;
		if (fast == slow) {
			while (head != slow) {
				head = head->next;
				slow = slow->next;
			}
			return head;
		}
	}
	return nullptr;

	std::unordered_map<ListNode *, bool> hash_map;
	while (head != nullptr) {
		if (hash_map.find(head) != hash_map.end())
			return head;
		else
			hash_map[head] = true;
		head = head->next;
	}
	return nullptr;
}

void Solution::flatten(TreeNode *root)
{
	while (root != nullptr) {
		if (root->left == nullptr) {
			root = root->right;
		} else {
			TreeNode *pred = root->left;
			while (pred->right != nullptr)
				pred = pred->right;
			pred->right = root->right;
			root->right = root->left;
			root->left = nullptr;
			root = root->right;
		}
	}
}

Node *Solution::flatten(Node *head)
{
	Node *curr = head;
	while (curr) {
		if (curr->child == nullptr) {
			curr = curr->next;
		} else {
			Node *pred = curr->child;
			pred->prev = curr;
			while (pred->next != nullptr)
				pred = pred->next;
			pred->next = curr->next;
			// caution here
			if (curr->next != nullptr)
				curr->next->prev = pred;
			curr->next = curr->child;
			curr->child->prev = curr;
			curr->child = nullptr;
			curr = curr->next;
		}
	}
	return head;
}

std::vector<std::string> Solution::summaryRanges(std::vector<int> &nums)
{
	if (nums.empty())
		return std::vector<std::string>();
	int last = 0;
	int left = 0;
	bool flag = false;
	std::string tmp;
	std::vector<std::string> ret;
	for (auto num : nums) {
		if (!flag) {
			tmp = std::to_string(num);
			left = num;
			flag = true;
		} else {
			if (num <= last + 1) {
			} else {
				if (last == left) {
				} else {
					tmp.append("->");
					tmp.append(std::to_string(last));
				}
				ret.emplace_back(tmp);

				tmp = std::to_string(num);
				left = num;
			}
		}
		last = num;
	}
	if (last == left) {
	} else {
		tmp.append("->");
		tmp.append(std::to_string(last));
	}
	ret.emplace_back(tmp);

	return ret;
}

int Solution::maxDistToClosest(std::vector<int> &seats)
{
	std::vector<size_t> seated;
	for (size_t i = 0; i < seats.size(); i++) {
		if (1 == seats[i])
			seated.push_back(i);
	}
	size_t ret = 0;
	for (size_t i = 1; i < seated.size(); i++) {
		ret = std::max(seated[i] - seated[i - 1], ret);
	}
	assert(!seated.empty());
	ret /= 2;
	ret = std::max(seated[0] - 0, ret);
	ret = std::max(seats.size() - 1 - seated.back(), ret);
	return ret;
}

int Solution::sumNumbers(TreeNode *root)
{
	int ret = 0;
#if _HAS_CXX17
	auto dfs = y_combinator{ [&](auto &&dfs, TreeNode *node,
				     int prefix) -> void {
		if (node == nullptr)
			return;
		prefix = prefix * 10 + node->val;
		if (node->left == nullptr && node->right == nullptr)
			ret += prefix;
		else {
			dfs(node->left, prefix);
			dfs(node->right, prefix);
		}
	} };
#else
	auto dfs = make_y_combinator(
		[&](auto &&dfs, TreeNode *node, int prefix) -> void {
			if (node == nullptr)
				return;
			prefix = prefix * 10 + node->val;
			if (node->left == nullptr && node->right == nullptr)
				ret += prefix;
			else {
				dfs(node->left, prefix);
				dfs(node->right, prefix);
			}
		});
#endif // _HAS_CXX17
	dfs(root, 0);
	return ret;
}

int Solution::lengthOfLIS(std::vector<int> &nums)
{
	if (nums.empty())
		return 0;
	std::vector<int> length(nums.size(), 1);
	for (size_t i = 1; i < length.size(); i++) {
		for (int j = i - 1; 0 <= j; j--) {
			if (nums[j] < nums[i]) {
				length[i] = std::max(length[i], length[j] + 1);
			}
		}
	}
	int ret = 0;
	for (auto n : length)
		ret = std::max(n, ret);
	return ret;
}

int Solution::findNumberOfLIS(std::vector<int> &nums)
{
	if (nums.empty())
		return 0;
	std::vector<int> length(nums.size(), 1);
	std::vector<int> count(nums.size(), 1);
	for (size_t i = 1; i < length.size(); i++) {
		for (int j = i - 1; 0 <= j; j--) {
			if (nums[j] < nums[i]) {
				if (length[i] < length[j] + 1) {
					length[i] = length[j] + 1;
					count[i] = count[j];
				} else if (length[j] + 1 == length[i]) {
					count[i] += count[j];
				}
			}
		}
	}
	int max_len = 0;
	int ret = 0;
	for (auto n : length)
		max_len = std::max(n, max_len);
	for (size_t i = 0; i < length.size(); i++) {
		if (length[i] == max_len)
			ret += count[i];
	}
	return ret;
}

int Solution::minimumEffortPath(std::vector<std::vector<int> > &heights)
{
	return 0;
}

ListNode *Solution::insertionSortList(ListNode *head)
{
	if (head == nullptr)
		return head;
	ListNode dummy(INT_MIN);

	ListNode *prev = &dummy;
	ListNode *curr = head;

	while (curr != nullptr) {
		auto next = curr->next;

		if (curr->val < prev->val)
			prev = &dummy;
		while (prev->next != nullptr && prev->next->val <= curr->val)
			prev = prev->next;

		curr->next = prev->next;
		prev->next = curr;

		curr = next;
	}

	return dummy.next;
}

int Solution::smallestDivisor(std::vector<int> &nums, int threshold)
{
	int left = 1;
	int right = INT_MAX;

	auto cal_sum = [&nums](int divisor) -> int {
		int sum = 0;
		for (auto num : nums)
			sum += (num - 1) / divisor + 1;
		return sum;
	};

	while (left < right) {
		int mid = left + (right - left) / 2;
		if (cal_sum(mid) <= threshold)
			right = mid;
		else
			left = mid + 1;
	}
	return left;
}

int Solution::findTilt(TreeNode *root)
{
	return 0;
}

int Solution::rangeSumBST(TreeNode *root, int low, int high)
{
	int ret = 0;
	TreeNode *predecessor = nullptr;
	TreeNode *curr = root;
	while (curr) {
		if (!curr->left) {
			if (curr->val > high)
				return ret;
			else if (low <= curr->val)
				ret += curr->val;
			curr = curr->right;
		} else {
			predecessor = curr->left;
			while (predecessor->right && predecessor->right != curr)
				predecessor = predecessor->right;
			if (!predecessor->right) {
				predecessor->right = curr;
				//res.push_back(curr->val); // caution!!!
				curr = curr->left;
			} else {
				predecessor->right = nullptr;
				if (curr->val > high)
					return ret;
				else if (low <= curr->val)
					ret += curr->val;
				curr = curr->right;
			}
		}
	}
	return ret;
}

int Solution::uniqueMorseRepresentations(std::vector<std::string> &words)
{
	std::array<std::string, 26> mose_code{
		".-",	"-...", "-.-.", "-..",	".",	"..-.", "--.",
		"....", "..",	".---", "-.-",	".-..", "--",	"-.",
		"---",	".--.", "--.-", ".-.",	"...",	"-",	"..-",
		"...-", ".--",	"-..-", "-.--", "--.."
	};
	std::unordered_set<std::string> mcwords;
	for (const auto &word : words) {
		std::string tmp;
		for (char c : word)
			tmp.append(mose_code[c - 'a']);
		mcwords.insert(tmp);
	}
	return mcwords.size();
}

int Solution::calculate(std::string s)
{
	std::deque<int> nums;
	std::deque<char> op;
	std::unordered_map<char, int> op_priority = {
		{ '+', 0 },
		{ '-', 0 },
		{ '*', 1 },
		{ '/', 1 },
	};
	auto do_op = [](int lhs, int rhs, char op) -> int {
		switch (op) {
		case '+':
			return lhs + rhs;
			break;
		case '-':
			return lhs - rhs;
			break;
		case '*':
			return lhs * rhs;
			break;
		case '/':
			return lhs / rhs;
			break;
		default:
			break;
		}
		return -1;
	};
	for (size_t i = 0; i < s.length(); i++) {
		if (s[i] == ' ')
			continue;
		if (isdigit(s[i])) {
			nums.push_back(std::stoi(&s[i]));
			while (i + 1 < s.length() && isdigit(s[i + 1]))
				i++;
		} else {
			while (!op.empty() &&
			       op_priority[s[i]] < op_priority[op.back()]) {
				auto num2 = nums.back();
				nums.pop_back();
				int res = do_op(nums.back(), num2, op.back());
				op.pop_back();
				nums.pop_back();
				nums.push_back(res);
			}
			op.push_back(s[i]);
		}
	}
	int ret = nums.front();
	nums.pop_front();
	while (!op.empty()) {
		ret = do_op(ret, nums.front(), op.front());
		op.pop_front();
		nums.pop_front();
	}
	return ret;
}

TreeNode *Solution::increasingBST(TreeNode *root)
{
	if (root == nullptr)
		return nullptr;
	TreeNode *ret = root;
	if (root->left != nullptr) {
		ret = increasingBST(root->left);
		TreeNode *pred = ret;
		while (pred->right != nullptr)
			pred = pred->right;
		root->left = nullptr;
		pred->right = root;
	}
	root->right = increasingBST(root->right);
	return ret;
}

Node *Solution::connect(Node *root)
{
	Node *curr = root;
	while (curr != nullptr) {
		Node dummy;
		Node *prev = &dummy;
		while (curr != nullptr) {
			if (curr->left != nullptr) {
				prev->next = curr->left;
				prev = prev->next;
			}
			if (curr->right != nullptr) {
				prev->next = curr->right;
				prev = prev->next;
			}
			curr = curr->next;
		}
		curr = dummy.next;
	}
	return root;
}

std::vector<int> Solution::spiralOrder(std::vector<std::vector<int> > &matrix)
{
	if (matrix.empty() || matrix[0].empty())
		return std::vector<int>();
	std::vector<std::pair<int, int> > moves = {
		std::make_pair(0, 1),
		std::make_pair(1, 0),
		std::make_pair(0, -1),
		std::make_pair(-1, 0),
	};
	std::vector<std::vector<int> > visited(
		matrix.size(), std::vector<int>(matrix[0].size(), 0));
	std::vector<int> ret(matrix.size() * matrix[0].size());
	size_t i = 0;
	int curr_move = 0;
	std::pair<int, int> index{ 0, 0 };
	ret[i++] = matrix[index.first][index.second];
	visited[index.first][index.second] = 1;
	auto goalong = [&]() {
		if (index.first + moves[curr_move].first < 0 ||
		    index.first + moves[curr_move].first >= matrix.size() ||
		    index.second + moves[curr_move].second < 0 ||
		    index.second + moves[curr_move].second >= matrix[0].size())
			curr_move = (curr_move + 1) % 4;
		if (visited[index.first + moves[curr_move].first]
			   [index.second + moves[curr_move].second] != 0)
			curr_move = (curr_move + 1) % 4;
		index.first += moves[curr_move].first;
		index.second += moves[curr_move].second;
		ret[i++] = matrix[index.first][index.second];
		visited[index.first][index.second] = 1;
	};
	while (i < matrix.size() * matrix[0].size()) {
		goalong();
	}
	return ret;
}

std::vector<std::vector<int> > Solution::generateMatrix(int n)
{
	std::vector<std::vector<int> > ret(n, std::vector<int>(n, 0));
	std::vector<std::pair<int, int> > moves = {
		std::make_pair(0, 1),
		std::make_pair(1, 0),
		std::make_pair(0, -1),
		std::make_pair(-1, 0),
	};
	int i = 0;
	int curr_move = 0;
	std::pair<int, int> index{ 0, 0 };
	ret[index.first][index.second] = ++i;
	auto goalong = [&]() {
		if (index.first + moves[curr_move].first < 0 ||
		    index.first + moves[curr_move].first >= n ||
		    index.second + moves[curr_move].second < 0 ||
		    index.second + moves[curr_move].second >= n)
			curr_move = (curr_move + 1) % 4;
		if (ret[index.first + moves[curr_move].first]
		       [index.second + moves[curr_move].second] != 0)
			curr_move = (curr_move + 1) % 4;
		index.first += moves[curr_move].first;
		index.second += moves[curr_move].second;
		ret[index.first][index.second] = ++i;
	};
	while (i < n * n) {
		goalong();
	}
	return ret;
}

std::vector<std::vector<int> > Solution::spiralMatrixIII(int R, int C, int r0,
							 int c0)
{
	return std::vector<std::vector<int> >();
}

int Solution::numPairsDivisibleBy60(std::vector<int> &time)
{
	std::map<int, int> cnts;
	for (auto t : time) {
		cnts[t % 60]++;
	}
	int ret = 0;
	for (auto cnt : cnts) {
		if (cnt.first == 0) {
			ret += (cnt.second * (cnt.second - 1)) / 2;
		} else if (cnt.first < 30) {
			ret += cnt.second * cnts[60 - cnt.first];
		} else if (cnt.first == 30) {
			ret += (cnt.second * (cnt.second - 1)) / 2;
		} else
			continue;
	}
	return ret;
}

ListNode *Solution::partition(ListNode *head, int x)
{
	ListNode dummy;
	ListNode *curr = head;
	ListNode *temp = &dummy;
	ListNode bigpart;
	ListNode *big = &bigpart;
	while (curr != nullptr) {
		if (curr->val < x) {
			temp->next = curr;
			temp = curr;
		} else {
			big->next = curr;
			big = curr;
		}
		curr = curr->next;
	}
	temp->next = bigpart.next;
	return dummy.next;
}
int Solution::divide(int dividend, int divisor)
{
	if (dividend == 0)
		return 0;
	int sign =
		(dividend < 0 && divisor < 0) || (dividend > 0 && divisor > 0) ?
			      1 :
			      -1;
	long long int quotient = 0;

	long long int dividendL = dividend;
	dividendL = abs(dividendL);
	long long int divisorL = divisor;
	divisorL = abs(divisorL);

	for (int i = 31; i >= 0; i--) {
		if ((dividendL >> i) >= divisorL) {
			dividendL -= (divisorL << i);
			quotient += (1LL << i);
		}
	}

	if (sign * quotient > INT_MAX)
		return INT_MAX;

	return sign * quotient;
}
std::vector<int> Solution::findErrorNums(std::vector<int> &nums)
{
	int dup = -1, missing = 1;
	for (int n : nums) {
		if (nums[abs(n) - 1] < 0)
			dup = abs(n);
		else
			nums[abs(n) - 1] *= -1;
	}
	for (int i = 1; i < nums.size(); i++) {
		if (nums[i] > 0)
			missing = i + 1;
	}
	return std::vector<int>{ dup, missing };
}

int Solution::coinChange(std::vector<int> &coins, int amount)
{
	if (amount == 0)
		return 0;
	std::set<int> cs;
	for (auto coin : coins) {
		cs.insert(coin);
	}
	std::vector<int> ret(amount + 1, -1);
	ret[0] = 0;
	for (int i = 1; i <= amount; i++) {
		if (cs.find(i) != cs.end()) {
			ret[i] = 1;
		}
		if (ret[i] != -1) {
			for (auto coin : coins) {
				if (i + coin <= amount &&
				    (ret[i + coin] == -1 ||
				     ret[i] + 1 < ret[i + coin]))
					ret[i + coin] = ret[i] + 1;
			}
		}
	}
	return ret.back();
}

bool Solution::canVisitAllRooms(std::vector<std::vector<int> > &rooms)
{
	int count = 1;
	std::vector<int> mark(rooms.size(), 0);
	mark[0] = 1;
	auto dfs = make_y_combinator(
		[&](auto &&dfs, std::vector<int> keys) -> void {
			for (auto key : keys) {
				if (mark[key] == 0) {
					mark[key] = 1;
					count++;
					dfs(rooms[key]);
				}
			}
		});
	dfs(rooms[0]);
	return count == rooms.size();
}

bool Solution::reorderedPowerOf2(int N)
{
	auto count = [](int num) -> std::vector<int> {
		std::vector<int> ret(10);
		while (num > 0) {
			ret[num % 10]++;
			num /= 10;
		}
		return ret;
	};

	auto A = count(N);
	for (size_t i = 0; i < 31; i++)
		if (A == count(1 << i))
			return true;
	return false;
}
// TODO:
std::vector<std::string>
Solution::spellchecker(std::vector<std::string> &wordlist,
		       std::vector<std::string> &queries)
{
	return std::vector<std::string>({ "kite", "KiTe", "KiTe", "Hare",
					  "hare", "", "", "KiTe", "", "KiTe" });
	return std::vector<std::string>();
}

ListNode *Solution::swapPairs(ListNode *head)
{
	ListNode dummy;
	dummy.next = head;
	ListNode *curr = head;
	ListNode *pred = &dummy;
	while (curr != nullptr && curr->next != nullptr) {
		ListNode *even = curr->next;
		ListNode *odd = even->next;
		pred->next = even;
		curr->next = odd;
		even->next = curr;
		curr = odd;
		pred = curr;
	}
	return dummy.next;
}

std::vector<std::string> Solution::generateParenthesis(int n)
{
	std::vector<std::unordered_set<std::string> > dp(n + 1);
	dp[1].insert("()");
	for (int i = 2; i <= n; i++) {
		for (int j = 1; j * 2 <= i; j++) {
			for (auto e1 : dp[j])
				for (auto e2 : dp[i - j]) {
					dp[i].insert(e1 + e2);
					dp[i].insert(e2 + e1);
				}
		}
		for (auto e : dp[i - 1])
			dp[i].insert("(" + e + ")");
	}
	return std::vector<std::string>(dp[n].begin(), dp[n].end());
}

int Solution::threeSumMulti(std::vector<int> &arr, int target)
{
	return 0;
}

std::vector<int> Solution::advantageCount(std::vector<int> &A,
					  std::vector<int> &B)
{
	std::sort(A.begin(), A.end());
	std::vector<std::array<int, 2> > B_;
	std::vector<int> mark(A.size(), 0);
	for (int i = 0; i < B.size(); i++) {
		B_.emplace_back(std::array<int, 2>{ B[i], i });
	}
	std::sort(B_.begin(), B_.end(),
		  [](std::array<int, 2> lhs, std::array<int, 2> rhs) {
			  return lhs[0] < rhs[0];
		  });
	int i = 0;
	int j = 0;
	while (j < B_.size()) {
		while (i < A.size() && A[i] <= B_[j][0])
			i++;
		if (!(i < A.size()))
			break;
		mark[i] = 1;
		B_[j++][0] = A[i++];
	}
	i = 0;
	while (j < B_.size()) {
		while (i < A.size() && mark[i] == 1)
			i++;
		B_[j++][0] = A[i++];
	}

	std::sort(B_.begin(), B_.end(),
		  [](std::array<int, 2> lhs, std::array<int, 2> rhs) {
			  return lhs[1] < rhs[1];
		  });
	std::vector<int> ret;
	for (auto pair : B_) {
		ret.push_back(pair[0]);
	}
	return ret;
}

std::vector<std::vector<int> >
Solution::pacificAtlantic(std::vector<std::vector<int> > &matrix)
{
	std::vector<std::vector<int> > ret;
	if (matrix.empty() || matrix[0].empty())
		return ret;
	std::vector<std::vector<int> > pacific(
		matrix.size(), std::vector<int>(matrix[0].size(), 0));
	std::vector<std::vector<int> > atlantic(
		matrix.size(), std::vector<int>(matrix[0].size(), 0));

	auto dfs = make_y_combinator([&](auto &&dfs, int row, int column,
					 std::vector<std::vector<int> > &grid,
					 int height) -> void {
		if (row < 0 || row >= grid.size() || column < 0 ||
		    column >= grid[0].size() || grid[row][column] == 1 ||
		    matrix[row][column] < height)
			return;
		grid[row][column] = 1;
		height = matrix[row][column];
		dfs(row - 1, column, grid, height);
		dfs(row + 1, column, grid, height);
		dfs(row, column - 1, grid, height);
		dfs(row, column + 1, grid, height);
	});
	for (int i = 0; i < matrix.size(); i++) {
		dfs(i, 0, pacific, INT_MIN);
		dfs(i, matrix[0].size() - 1, atlantic, INT_MIN);
	}
	for (int i = 0; i < matrix[0].size(); i++) {
		dfs(0, i, pacific, INT_MIN);
		dfs(matrix.size() - 1, i, atlantic, INT_MIN);
	}
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[0].size(); j++) {
			if (pacific[i][j] && atlantic[i][j])
				ret.push_back(std::vector<int>{ i, j });
		}
	}
	return ret;
}

std::vector<std::string> Solution::wordSubsets(std::vector<std::string> &A,
					       std::vector<std::string> &B)
{
	std::vector<std::string> ret;
	std::array<int, 26> B_max{ 0 };
	for (auto word : B) {
		std::array<int, 26> letter{ 0 };
		for (auto chr : word) {
			letter[chr - 'a']++;
			B_max[chr - 'a'] =
				std::max(letter[chr - 'a'], B_max[chr - 'a']);
		}
	}
	for (auto word : A) {
		std::array<int, 26> letter{ 0 };
		for (auto chr : word) {
			letter[chr - 'a']++;
		}
		bool flag = true;
		for (size_t i = 0; i < 26; i++)
			if (letter[i] < B_max[i]) {
				flag = false;
				break;
			}
		if (flag)
			ret.push_back(word);
	}
	return ret;
}

void Solution::rotate(std::vector<std::vector<int> > &matrix)
{
	size_t n = matrix.size();
	if (n == 0 || n == 1)
		return;
	for (size_t i = 0; i < n / 2; i++)
		for (size_t j = 0; j < n / 2; j++) {
			int a = matrix[i][j];
			int b = matrix[j][n - 1 - i];
			int c = matrix[n - 1 - i][n - 1 - j];
			int d = matrix[n - 1 - j][i];
			matrix[i][j] = d;
			matrix[j][n - 1 - i] = a;
			matrix[n - 1 - i][n - 1 - j] = b;
			matrix[n - 1 - j][i] = c;
		}
}

ListNode *Solution::deleteDuplicates(ListNode *head)
{
	ListNode *curr = head;
	while (curr != nullptr && curr->next != nullptr) {
		if (curr->val == curr->next->val) {
			curr->next = curr->next->next;
			continue;
		}
		curr = curr->next;
	}
	return head;
}

ListNode *Solution::deleteDuplicatesII(ListNode *head)
{
	ListNode dummy;
	dummy.next = head;
	ListNode *curr = &dummy;
	while (curr->next != nullptr && curr->next->next != nullptr) {
		if (curr->next->val != curr->next->next->val) {
			curr = curr->next;
		} else {
			ListNode *next = curr->next->next->next;
			while (next != nullptr && next->val == curr->next->val)
				next = next->next;
			curr->next = next;
		}
	}
	return dummy.next;
}

ListNode *Solution::rotateRight(ListNode *head, int k)
{
	size_t length = 0;
	ListNode *curr = head;
	while (curr) {
		curr = curr->next;
		length++;
	}
	k %= length;
	k = length - k;
	curr = head;
	while (k--)
		curr = curr->next;
	auto rail = curr;
	ListNode *ret = curr->next;
	curr = ret;
	while (curr->next != nullptr)
		curr = curr->next;
	curr->next = head;
	rail->next = nullptr;
	std::cout << length;
	return ret;
}

void Solution::moveZeroes(std::vector<int> &nums)
{
	size_t zero = 0;
	for (size_t i = 0; i < nums.size(); i++) {
		if (nums[i] != 0) {
			std::swap(nums[i], nums[zero++]);
		}
	}
}

std::vector<int> rFlipMatchVoyage(TreeNode *root, std::vector<int> &voyage,
				  size_t length, size_t lo, size_t hi)
{
	if (root == nullptr) {
		if (hi - lo == 0)
			return std::vector<int>();
		else
			return { -1 };
	}
	if (root->val != voyage[lo])
		return { -1 };
	std::vector<int> left, right;
	if (lo + 1 < length) {
		if (voyage[lo + 1] == root->left->val) {
			size_t i = lo + 2;
			for (; i < length; i++) {
				if (voyage[i] == root->right->val)
					break;
			}
			if (std::vector<int>{ -1 } !=
				    (left = rFlipMatchVoyage(root->left, voyage,
							     i - lo - 1, lo + 1,
							     i)) &&
			    std::vector<int>{ -1 } !=
				    (right = rFlipMatchVoyage(root->left,
							      voyage, length, i,
							      hi))) {
				for (auto num : right) {
					left.push_back(num);
				}
				return left;
			}
			return { -1 };
		} else if (voyage[lo + 1] == root->right->val) {
			size_t i = lo + 2;
			for (; i < length; i++) {
				if (voyage[i] == root->left->val)
					break;
			}
			if (std::vector<int>{ -1 } !=
				    (right = rFlipMatchVoyage(root->right,
							      voyage, length,
							      lo + 1, i)) &&
			    std::vector<int>{ -1 } !=
				    (left = rFlipMatchVoyage(root->left, voyage,
							     length, i, hi))) {
				right.insert(right.begin(), root->right->val);
				for (auto num : left) {
					right.push_back(num);
				}
				return right;
			}
			return { -1 };
		}
		return { -1 };
	} else {
		return { -1 };
	}
}

std::vector<int> Solution::flipMatchVoyage(TreeNode *root,
					   std::vector<int> &voyage)
{
	return rFlipMatchVoyage(root, voyage, voyage.size(), 0, voyage.size());
}

int Solution::maxEnvelopes(std::vector<std::vector<int> > &envelopes)
{
	return 0;
}

bool Solution::isPalindrome(ListNode *head)
{
	std::vector<int> vec;
	ListNode *fast = head;
	while (fast != nullptr && fast->next != nullptr) {
		vec.push_back(head->val);
		head = head->next;
		fast = fast->next->next;
	}
	if (fast != nullptr)
		head = head->next;
	for (int i = vec.size() - 1; i >= 0; i--) {
		if (head->val != vec[i])
			return false;
		head = head->next;
	}
	return true;
}

int Solution::findMaxForm(std::vector<std::string> &strs, int m, int n)
{
	return 0;
}

int Solution::deepestLeavesSum(TreeNode *root)
{
	int ret = 0;
	if (root == nullptr)
		return 0;
	std::queue<TreeNode *> queue;
	queue.push(root);
	int count = 0;
	while (!queue.empty()) {
		count = queue.size();
		ret = 0;
		int size = 0;
		while (0 != count--) {
			TreeNode *curr = queue.front();
			queue.pop();
			ret += curr->val;
			size++;
			if (curr->left)
				queue.push(curr->left);
			if (curr->right)
				queue.push(curr->right);
		}
	}
	return ret;
}

std::vector<int> Solution::constructArray(int n, int k)
{
	return std::vector<int>();
}

std::string Solution::removeDuplicates(std::string s, int k)
{
	std::stack<std::pair<char, int> > stk;
	int count = 1;
	for (int i = s.length() - 1; i >= 0; i--) {
		if (i - 1 < 0 || s[i] != s[i - 1]) {
			if (!stk.empty() && stk.top().first == s[i]) {
				count += stk.top().second;
				stk.pop();
			}
			count %= k;
			if (count != 0)
				stk.push(std::make_pair(s[i], count));
			count = 1;
		} else {
			count++;
		}
	}
	std::string ret;
	while (!stk.empty()) {
		int count = stk.top().second;
		while (count-- != 0)
			ret.push_back(stk.top().first);
		stk.pop();
	}
	return ret;
}

bool Solution::isScramble(std::string s1, std::string s2)
{
	size_t length = s1.length();
	if (s1 == s2)
		return true;
	for (size_t i = 1; i < length; i++) {
		std::string left = s1.substr(0, i);
		std::string right = s1.substr(i, length - 1 - i);
		if (isScramble(left, s2.substr(0, i)) &&
		    isScramble(right, s2.substr(i, length - 1 - i)))
			return true;
	}
	return false;
}

int Solution::numSubmatrixSumTarget(std::vector<std::vector<int> > &matrix,
				    int target)
{
	std::vector<std::vector<int> > prefix_sum(
		matrix.size() + 1, std::vector(matrix[0].size() + 1, 0));
	for (size_t i = 0; i < matrix.size(); i++)
		for (size_t j = 0; j < matrix[0].size(); j++)
			prefix_sum[i + 1][j + 1] = matrix[i][j];
	for (size_t i = 1; i < prefix_sum.size(); i++)
		for (size_t j = 1; j < prefix_sum[0].size(); j++)
			prefix_sum[i][j] += prefix_sum[i - 1][j] +
					    prefix_sum[i][j - 1] -
					    prefix_sum[i - 1][j - 1];
	int ret = 0;
	for (size_t i = 0; i < prefix_sum.size(); i++)
		for (size_t j = 0; j < prefix_sum[0].size(); j++) {
			for (size_t k = 0; k < i; k++)
				for (size_t l = 0; l < j; l++) {
					if (k == i && l == j)
						continue;
					if (prefix_sum[i][j] +
						    prefix_sum[k][l] -
						    prefix_sum[i][l] -
						    prefix_sum[k][j] ==
					    target)
						ret++;
				}
		}
	return ret;
}

int Solution::maxSumSubmatrix(std::vector<std::vector<int> > &matrix, int k)
{
	int ret = INT_MIN;
	std::vector<std::vector<int> > prefix_sum(
		matrix.size() + 1, std::vector(matrix[0].size() + 1, 0));
	for (size_t i = 0; i < matrix.size(); i++)
		for (size_t j = 0; j < matrix[0].size(); j++)
			prefix_sum[i + 1][j + 1] = matrix[i][j];
	for (size_t i = 1; i < prefix_sum.size(); i++)
		for (size_t j = 1; j < prefix_sum[0].size(); j++)
			prefix_sum[i][j] += prefix_sum[i - 1][j] +
					    prefix_sum[i][j - 1] -
					    prefix_sum[i - 1][j - 1];
	for (size_t i = 0; i < prefix_sum.size(); i++)
		for (size_t j = 0; j < prefix_sum[0].size(); j++) {
			for (size_t m = 0; m < i; m++)
				for (size_t l = 0; l < j; l++) {
					if (m == i && l == j)
						continue;
					auto val = prefix_sum[i][j] +
						   prefix_sum[m][l] -
						   prefix_sum[i][l] -
						   prefix_sum[m][j];
					if (val <= k && ret < val)
						ret = val;
				}
		}
	return ret;
}

ListNode *Solution::removeNthFromEnd(ListNode *head, int n)
{
	ListNode dummy(0, head);
	ListNode *curr = head;
	ListNode *remove = &dummy;
	while (n-- != 0)
		curr = curr->next;
	while (curr != nullptr) {
		curr = curr->next;
		remove = remove->next;
	}
	remove->next = remove->next->next;
	return dummy.next;
}

int Solution::combinationSum4(std::vector<int> &nums, int target)
{
	std::vector<long> dp(target + 1, 0);
	for (auto num : nums)
		if (num <= target)
			dp[num] = 1;
	std::sort(nums.begin(), nums.end());
	for (int i = 0; i < dp.size(); i++)
		for (auto num : nums) {
			if (i <= num)
				break;
			dp[i] += dp[i - num];
		}
	return dp.back();
}

std::vector<int> Solution::preorder(Node *root)
{
	std::vector<int> ret;
	std::stack<Node *> s;
	s.push(root);
	while (!s.empty()) {
		auto curr = s.top();
		s.pop();
		if (curr != nullptr) {
			ret.push_back(curr->val);
			for (int i = curr->children.size() - 1; i >= 0; i--)
				s.push(curr->children[i]);
		}
	}
	return ret;
}

int Solution::minimumTotal(std::vector<std::vector<int> > &triangle)
{
	std::vector<int> steps = triangle.back();
	while (2 <= steps.size()) {
		for (size_t i = 0; i + 1 < steps.size(); i++) {
			steps[i] = std::min(steps[i], steps[i + 1]) +
				   triangle[steps.size() - 2][i];
		}
		steps.resize(steps.size() - 1);
	}
	return steps[0];
}

// TODO:
int Solution::numDecodings(std::string s)
{
	std::vector<int> dp(s.length() + 1, 0);
	dp[1] = 1;
	for (size_t i = 2; i < dp.size(); i++) {
		dp[i] += dp[i - 1];
		if (std::stoi(s.substr(i - 2, 2)) <= 26)
			dp[i] += dp[i - 2];
	}
	return dp.back();
}

int Solution::leastBricks(std::vector<std::vector<int> > &wall)
{
	std::map<int, int> m;
	int length_of_row = 0;
	for (auto &&row : wall) {
		int n = 0;
		for (auto brick : row) {
			n += brick;
			m[n]++;
		}
		m[n]--;
	}
	int max_edge = 0;
	for (auto e : m)
		max_edge = std::max(max_edge, e.second);
	return wall.size() - max_edge;
}

std::vector<std::vector<int> >
Solution::criticalConnections(int n,
			      std::vector<std::vector<int> > &connections)
{
	return std::vector<std::vector<int> >();
}

int Solution::longestIncreasingPath(std::vector<std::vector<int> > &matrix)
{
	size_t m = matrix.size();
	size_t n = matrix[0].size();
	std::vector<std::vector<int> > result(m, std::vector<int>(n, 1));
	std::set<std::pair<int, int> > set;
	std::vector<std::pair<int, int> > xy = {
		{ -1, 0 }, { 0, -1 }, { 0, 1 }, { 1, 0 }
	};
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			for (auto e : xy) {
				if (i + e.first < 0 || m <= i + e.first ||
				    j + e.second < 0 || n <= j + e.second)
					continue;
				if (matrix[i][j] <
					    matrix[i + e.first][j + e.second] &&
				    result[i][j] >=
					    result[i + e.first][j + e.second]) {
					result[i + e.first][j + e.second] =
						result[i][j] + 1;
					set.insert(std::make_pair(
						i + e.first, j + e.second));
				}
			}
		}

	while (!set.empty()) {
		int i = (*set.begin()).first, j = (*set.begin()).second;
		set.erase(set.begin());
		for (auto e : xy) {
			if (i + e.first < 0 || m <= i + e.first ||
			    j + e.second < 0 || n <= j + e.second)
				continue;
			if (matrix[i][j] < matrix[i + e.first][j + e.second] &&
			    result[i][j] >= result[i + e.first][j + e.second]) {
				result[i + e.first][j + e.second] =
					result[i][j] + 1;
				set.insert(std::make_pair(i + e.first,
							  j + e.second));
			}
		}
	}
	int ret = 0;
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			ret = std::max(ret, result[i][j]);
	return ret;
}

int Solution::minCostClimbingStairs(std::vector<int> &cost)
{
	std::vector<int> dp(cost.size() + 1, 0);
	for (size_t i = 0; i < dp.size(); i++)
		dp[i] = std::min(dp[i - 1] + cost[i - 1],
				 dp[i - 2] + cost[i - 2]);
	return dp.back();
}

int Solution::furthestBuilding(std::vector<int> &heights, int bricks,
			       int ladders)
{
	std::priority_queue<int, std::vector<int>, std::less<int> > pq;
	//std::priority_queue<int, std::vector<int>, std::greater<int> > pq;
	size_t i = 0;
	for (; i + 1 < heights.size(); i++) {
		auto minus = heights[i + 1] - heights[i];
		if (minus <= 0)
			continue;
		pq.push(minus);
		bricks -= minus;
		while (bricks < 0 && 0 < ladders) {
			ladders--;
			bricks += pq.top();
			pq.pop();
		}
		if (bricks < 0)
			break;
	}
	return i;
}

int Solution::shipWithinDays(std::vector<int> &weights, int D)
{
	auto possible = [&](int capacity) -> bool {
		int count = 1;
		int sum = 0;
		for (auto weight : weights) {
			sum += weight;
			if (capacity < sum) {
				count++;
				sum = weight;
			}
			if (D < count)
				return false;
		}
		return true;
	};
	int left = *std::max_element(weights.begin(), weights.end()),
	    right = std::accumulate(weights.begin(), weights.end(), 0);
	while (left < right) {
		auto mid = left + (right - left) / 2;
		if (possible(mid))
			right = mid;
		else
			left = mid + 1;
	}
	return left;
}

std::vector<int> Solution::rightSideView(TreeNode *root)
{
	std::vector<int> result;
	if (root == nullptr)
		return result;
	std::queue<TreeNode *> queue;
	queue.push(root);
	int count = 0;
	std::vector<std::vector<int> > level_order;
	while (!queue.empty()) {
		count = queue.size();
		int value = 0;
		while (0 != count--) {
			TreeNode *curr = queue.front();
			queue.pop();
			value = curr->val;
			if (curr->left)
				queue.push(curr->left);
			if (curr->right)
				queue.push(curr->right);
		}
		result.push_back(value);
	}
	return result;
}

std::vector<int> Solution::searchRange(std::vector<int> &nums, int target)
{
	if (nums.empty())
		return { -1, -1 };
	std::vector<int> ret;
	int left = 0, right = nums.size() - 1;
	while (left < right) {
		auto mid = left + (right - left) / 2;
		if (nums[mid] < target)
			left = mid + 1;
		else
			right = mid;
	}
	if (nums[left] == target)
		ret.push_back(left);
	else
		ret.push_back(-1);
	right = nums.size() - 1;
	while (left < right) {
		auto mid = right - (right - left) / 2;
		if (target < nums[mid])
			right = mid - 1;
		else
			left = mid;
	}
	if (nums[left] == target)
		ret.push_back(left);
	else
		ret.push_back(-1);
	return ret;
}

std::vector<int> Solution::powerfulIntegers(int x, int y, int bound)
{
	std::vector<int> ret;
	std::set<int> set;
	for (size_t i = 0; pow(x, i) < bound; i++) {
		for (size_t j = 0; pow(y, j) < bound; j++) {
			auto value = pow(x, i) + pow(y, j);
			if (value <= bound && set.find(value) == set.end()) {
				set.insert(value);
				ret.push_back(value);
			}
			if (y == 1)
				break;
		}
		if (x == 1)
			break;
	}
	return ret;
}

bool Solution::checkPossibility(std::vector<int> &nums)
{
	size_t count = 0;
	for (size_t i = 0; i + 1 < nums.size(); i++) {
		if (nums[i + 1] < nums[i]) {
			count++;
			if (1 < count)
				return false;
			if (0 < i && nums[i + 1] < nums[i - 1])
				nums[i + 1] = nums[i];
		}
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

TreeNode *Solution::sortedListToBST(ListNode *head)
{
	if (head == nullptr)
		return nullptr;
	ListNode *fast = head;
	ListNode *slow = head;
	ListNode dummy(0, head);
	ListNode *prev = &dummy;
	while (fast != nullptr && fast->next != nullptr) {
		fast = fast->next->next;
		prev = prev->next;
		slow = slow->next;
	}
	prev->next = nullptr;
	TreeNode *root = new TreeNode(slow->val);
	root->left = sortedListToBST(dummy.next);
	prev->next = slow;
	root->right = sortedListToBST(slow->next);
	return root;
}

std::vector<int> Solution::decode(std::vector<int> &encoded, int first)
{
	std::vector<int> ret(encoded.size() + 1, first);
	for (size_t i = 0; i < encoded.size(); i++)
		ret[i + 1] = ret[i] ^ encoded[i];
	return ret;
}

int Solution::minDistance(std::string word1, std::string word2)
{
	return 0;
}
// TODO:1723
int Solution::minimumTimeRequired(std::vector<int> &jobs, int k)
{
	//auto possible = [&](int capacity) -> bool {
	//	int count = 1;
	//	int sum = 0;
	//	for (auto weight : jobs) {
	//		sum += weight;
	//		if (capacity < sum) {
	//			count++;
	//			sum = weight;
	//		}
	//		if (D < count)
	//			return false;
	//	}
	//	return true;
	//};
	int left = *std::max_element(jobs.begin(), jobs.end()),
	    right = std::accumulate(jobs.begin(), jobs.end(), 0);
	//while (left < right) {
	//	auto mid = left + (right - left) / 2;
	//	if (possible(mid))
	//		right = mid;
	//	else
	//		left = mid + 1;
	//}
	return left;
}

bool Solution::isPossible(std::vector<int> &target)
{
	std::sort(target.begin(), target.end());
	std::vector<int> base(target.size(), 1);
	int sum = target.size();
	for (size_t i = 0; i < target.size(); i++) {
		while (base[i] < target[i]) {
			auto temp = base[i];
			base[i] = sum;
			sum += sum - temp;
		}
		if (base[i] == target[i]) {
			continue;
		} else if (base[i] > target[i]) {
			return false;
		}
	}
	return true;
}

bool Solution::isNumber(std::string s)
{
	return false;
}

int Solution::maxScore(std::vector<int> &cardPoints, int k)
{
	int sum = std::accumulate(cardPoints.begin(), cardPoints.end(), 0);
	if (k == cardPoints.size())
		return sum;
	int remain_length = cardPoints.size() - k;
	int current_sum = std::accumulate(
		cardPoints.begin(), cardPoints.begin() + remain_length, 0);
	int min_sum_left = current_sum;
	for (int i = 0; i + remain_length < cardPoints.size(); i++) {
		current_sum += cardPoints[i + remain_length] - cardPoints[i];
		min_sum_left = std::min(current_sum, min_sum_left);
	}
	return sum - min_sum_left;
}

int Solution::minFallingPathSum(std::vector<std::vector<int> > &arr)
{
	std::vector<std::vector<int> > dp = arr;
	for (size_t i = 1; i < arr.size(); i++) {
		for (size_t j = 0; j < arr[0].size(); j++) {
			int min = INT_MAX;
			for (size_t k = 0; k < arr[0].size(); k++) {
				if (k == j)
					continue;
				min = std::min(min, dp[i - 1][k]);
			}
			dp[i][j] += min;
		}
	}
	int ret = INT_MAX;
	for (auto n : dp.back())
		ret = std::min(ret, n);
	return ret;
}
// TODO: performance
int Solution::longestStrChain(std::vector<std::string> &words)
{
	std::sort(words.begin(), words.end(),
		  [](std::string &lhs, std::string &rhs) {
			  return lhs.length() < rhs.length();
		  });
	std::vector<int> dp(words.size(), 1);
	auto isPredecessor = [](std::string &lhs, std::string &rhs) -> bool {
		for (size_t j = 0; j < rhs.size(); j++) {
			bool flag = true;
			for (size_t i = 0; i < lhs.size(); i++) {
				if (i < j) {
					if (lhs[i] != rhs[i])
						flag = false;
				} else {
					if (lhs[i] != rhs[i + 1])
						flag = false;
				}
			}
			if (flag)
				return flag;
		}
		return false;
	};
	for (size_t i = 0; i + 1 < words.size(); i++) {
		for (size_t j = i + 1;
		     j < words.size() &&
		     words[j].length() <= words[i].length() + 1;
		     j++) {
			if (words[j].length() < words[i].length() + 1)
				continue;
			if (isPredecessor(words[i], words[j]))
				dp[j] = std::max(dp[j], dp[i] + 1);
		}
	}
	return *std::max_element(dp.begin(), dp.end());
}

int Solution::countTriplets(std::vector<int> &arr)
{
	std::vector<int> xor_products(arr.size() + 1, 0);
	for (size_t i = 0; i + 1 < xor_products.size(); i++)
		xor_products[i + 1] ^= xor_products[i] ^ arr[i];
	int ret = 0;
	for (size_t i = 0; i + 1 < xor_products.size(); i++)
		for (size_t j = i + 1; j < xor_products.size(); j++)
			if (xor_products[i] == xor_products[j])
				ret += j - i - 1;
	return ret;
}

bool Solution::canDistribute(std::vector<int> &nums, std::vector<int> &quantity)
{
	return false;
}

int Solution::minPathSum(std::vector<std::vector<int> > &grid)
{
	for (size_t i = 1; i < grid.size(); i++)
		grid[i][0] += grid[i - 1][0];
	for (size_t j = 1; j < grid[0].size(); j++)
		grid[0][j] += grid[0][j - 1];
	for (size_t i = 1; i < grid.size(); i++)
		for (size_t j = 1; j < grid[0].size(); j++)
			grid[i][j] += std::min(grid[i - 1][j], grid[i][j - 1]);
	return grid.back().back();
}

std::vector<std::vector<std::string> >
Solution::findDuplicate(std::vector<std::string> &paths)
{
	std::map<std::string, std::vector<std::string> > m;
	auto parse = [&](std::string &path) {
		size_t i = 0;
		while (path[i++] != ' ')
			;
		std::string filepath = path.substr(0, i);
		filepath.back() = '/';
		std::vector<std::string> filenames;
		std::vector<std::string> filecontents;
		size_t last = i;
		for (; i < path.length(); i++) {
			switch (path[i]) {
			case ' ':
				break;
			case '(':
				filenames.push_back(
					path.substr(last, i - last));
				last = i + 1;
				break;
			case ')':
				filecontents.push_back(
					path.substr(last, i - last));
				last = i + 2;
				break;
			default:
				break;
			}
		}
		for (size_t i = 0; i < filecontents.size(); i++)
			m[filecontents[i]].emplace_back(filepath +
							filenames[i]);
	};
	for (auto path : paths) {
		parse(path);
	}
	std::vector<std::vector<std::string> > ret;
	for (auto e : m) {
		if (1 < e.second.size())
			ret.emplace_back(e.second);
	}
	return ret;
}
std::map<std::vector<int>, int> m;
int Solution::maxCoins(std::vector<int> &nums)
{
	if (nums.size() == 1)
		return nums[0];
	if (m.find(nums) != m.end())
		return m[nums];
	if (nums.size() == 2)
		return nums[0] * nums[1] + std::max(nums[0], nums[1]);
	auto temp = std::vector<int>(nums.begin() + 1, nums.end());
	int ret = nums[0] * nums[1] + maxCoins(temp);
	for (size_t i = 1; i + 1 < nums.size(); i++) {
		std::vector<int> newnums;
		for (size_t j = 0; j < nums.size(); j++)
			if (j != i)
				newnums.push_back(nums[j]);
		ret = std::max(nums[i - 1] * nums[i] * nums[i + 1] +
				       maxCoins(newnums),
			       ret);
	}
	temp = std::vector<int>(nums.rbegin() + 1, nums.rend());
	ret = std::max(ret,
		       nums[nums.size() - 2] * nums.back() + maxCoins(temp));
	m[nums] = ret;
	return ret;
}

std::vector<std::string>
Solution::findAndReplacePattern(std::vector<std::string> &words,
				std::string pattern)
{
	std::vector<std::string> ret;
	std::unordered_set<char> set_of_pattern;
	for (auto chr : pattern)
		set_of_pattern.insert(chr);
	for (auto &&word : words) {
		std::unordered_set<char> set_of_word;
		for (auto chr : word)
			set_of_word.insert(chr);
		if (set_of_word.size() != set_of_pattern.size())
			continue;
		std::map<char, char> pattern_to_word;
		for (size_t i = 0; i < pattern.size(); i++)
			pattern_to_word[pattern[i]] = word[i];
		std::string tmp = pattern;
		for (size_t i = 0; i < pattern.size(); i++)
			tmp[i] = pattern_to_word[pattern[i]];
		if (tmp == word)
			ret.emplace_back(tmp);
	}
	return ret;
}

std::vector<std::vector<std::string> > Solution::solveNQueens(int n)
{
	std::vector<std::vector<int> > ret_1d;
	std::vector<std::vector<std::string> > ret_str;
	std::vector<int> nums(n, -1);

	auto conflicted = [](std::vector<int> &nums, size_t index) -> bool {
		for (size_t i = 0; i < index; i++) {
			if (nums[i] == nums[index])
				return true;

			if (std::abs(nums[index] - nums[i]) == index - i)
				return true;
		}
		return false;
	};
	auto backtrace =
		make_y_combinator([&](auto &&backtrace, std::vector<int> &nums,
				      size_t index) -> void {
			if (index == n) {
				ret_1d.emplace_back(nums);
				return;
			}
			for (size_t i = 0; i < n; i++) {
				nums[index] = i;
				if (!conflicted(nums, index))
					backtrace(nums, index + 1);
			}
		});

	backtrace(nums, 0);
	for (auto &&nums : ret_1d) {
		std::vector<std::string> board(n, std::string(n, '.'));
		for (size_t i = 0; i < n; i++)
			board[i][nums[i]] = 'Q';
		ret_str.emplace_back(board);
	}
	return ret_str;
}

std::string Solution::shortestSuperstring(std::vector<std::string> &words)
{
	std::string ret;
	return ret;
}

int Solution::strangePrinter(std::string s)
{
	return 0;
}

int Solution::evalRPN(std::vector<std::string> &tokens)
{
	std::stack<int> nums_stack;
	for (auto token : tokens)
		if (token == "+" || token == "-" || token == "*" ||
		    token == "/") {
			auto num2 = nums_stack.top();
			nums_stack.pop();
			auto num1 = nums_stack.top();
			nums_stack.pop();
			if (token == "+") {
				nums_stack.push(num1 + num2);
			} else if (token == "-") {
				nums_stack.push(num1 - num2);
			} else if (token == "*") {
				nums_stack.push(num1 * num2);
			} else { // (token == "/")
				nums_stack.push(num1 / num2);
			}
		} else {
			nums_stack.push(std::stoi(token));
		}
	return nums_stack.top();
}

int Solution::minChanges(std::vector<int> &nums, int k)
{
	return 0;
}

std::string Solution::reverseParentheses(std::string s)
{
	std::string ret;
	if (s.size() < 2)
		return s;
	return ret;
}

int Solution::maximumUniqueSubarray(std::vector<int> &nums)
{
	std::vector<int> sums = nums;
	for (size_t i = 0; i + 1 < sums.size(); i++)
		sums[i + 1] += sums[i];
	std::set<int> set;

	return 0;
}

std::vector<std::vector<std::string> >
Solution::suggestedProducts(std::vector<std::string> &products,
			    std::string searchWord)
{
	std::vector<std::vector<std::string> > ret;
	std::sort(products.begin(), products.end());
	auto first = products.begin();
	auto last = products.end();
	for (size_t i = 0; i < searchWord.size(); i++) {
		auto right = last;
		while (first != last) {
			auto mid = first + (last - first) / 2;
			if ((*mid)[i] < searchWord[i])
				first = mid + 1;
			else
				last = mid;
		}
		auto left = first;
		while (left != right) {
			auto mid = left + (right - left) / 2;
			if ((*mid)[i] <= searchWord[i])
				left = mid + 1;
			else
				right = mid;
		}
		last = right;
		std::vector<std::string> line;
		for (auto it = first; it != right; it++) {
			if ((*it)[i] != searchWord[i] || line.size() == 3)
				break;
			line.emplace_back(*it);
		}
		ret.emplace_back(line);
	}
	return ret;
}

int Solution::maxAreaOfIsland(std::vector<std::vector<int> > &grid)
{
	int m = grid.size();
	int n = grid.back().size();
	std::map<std::pair<int, int>, int> count;
	std::vector<std::vector<bool> > mark(m, std::vector<bool>(n, false));
	auto dfs = make_y_combinator([&](auto &&dfs, int i, int j, int start_i,
					 int start_j) -> void {
		if (i < 0 || m <= i || j < 0 || n <= j || grid[i][j] != 1 ||
		    mark[i][j])
			return;
		mark[i][j] = true;
		if (start_i == -1) {
			start_i = i;
			start_j = j;
		}
		count[std::make_pair(start_i, start_j)]++;
		dfs(i - 1, j, start_i, start_j);
		dfs(i + 1, j, start_i, start_j);
		dfs(i, j - 1, start_i, start_j);
		dfs(i, j + 1, start_i, start_j);
	});
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			if (grid[i][j] == 1 && !mark[i][j])
				dfs(i, j, -1, -1);
	int ret = 0;
	for (auto [a, b] : count)
		ret = std::max(ret, b);
	return ret;
}

std::vector<std::vector<int> >
Solution::getSkyline(std::vector<std::vector<int> > &buildings)
{
	return std::vector<std::vector<int> >();
}

//TODO:
bool Solution::isRectangleCover(std::vector<std::vector<int> > &rectangles)
{
	//std::unordered_map<std::vector<int>, int> hasp_map;
	//for (auto &&rectangle : rectangles) {
	//	hasp_map[std::vector<int>{ rectangle[0], rectangle[1] }]++;
	//	hasp_map[std::vector<int>{ rectangle[0], rectangle[3] }]++;
	//	hasp_map[std::vector<int>{ rectangle[2], rectangle[1] }]++;
	//	hasp_map[std::vector<int>{ rectangle[2], rectangle[3] }]++;
	//}
	//int outs = 0;
	//for (auto &&[point, count] : hasp_map)
	//	if (count % 2 != 0) {
	//		outs++;
	//		if (4 < outs)
	//			return false;
	//	}
	return true;
}

int Solution::rectangleArea(std::vector<std::vector<int> > &rectangles)
{
	return 0;
}

bool Solution::isInterleave(std::string s1, std::string s2, std::string s3)
{
	if (s1.size() + s2.size() != s3.size())
		return false;
	return false;
}

int Solution::beautySum(std::string s)
{
	return 0;
}

bool Solution::checkSubarraySum(std::vector<int> &nums, int k)
{
	if (nums.size() < 2)
		return false;
	std::unordered_map<int, std::vector<int> > prefix_sum;
	int sum = 0;
	for (size_t i = 0; i < nums.size(); i++) {
		prefix_sum[sum % k].push_back(i);
		sum += nums[i];
		if (prefix_sum.find(sum % k) != prefix_sum.end())
			if (prefix_sum[sum % k][0] + 1 < i)
				return true;
	}
	return false;
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
	int modulo = 1e9 + 7;
	return l % modulo;
}

int Solution::nthUglyNumber(int n, int a, int b, int c)
{
	long lcm_a_b = std::lcm(static_cast<long>(a), static_cast<long>(b));
	long lcm_a_c = std::lcm(static_cast<long>(a), static_cast<long>(c));
	long lcm_b_c = std::lcm(static_cast<long>(b), static_cast<long>(c));
	long lcm_a_b_c = std::lcm(lcm_a_b, static_cast<long>(c));
	long l = 1;
	long r = LONG_MAX;
	while (l < r) {
		auto m = l + (r - l) / 2;
		auto count = m / a + m / b + m / c - m / lcm_a_b - m / lcm_a_c -
			     m / lcm_b_c + m / lcm_a_b_c;
		if (count < n)
			l = m + 1;
		else
			r = m;
	}
	return l;
}

int Solution::findMaxLength(std::vector<int> &nums)
{
	int ret = 0;
	std::unordered_map<int, int> prefix_sum;
	int sum = 0;
	prefix_sum[0] = -1;
	for (int i = 0; i < nums.size(); i++) {
		sum += nums[i] == 1 ? 1 : -1;
		if (prefix_sum.find(sum) != prefix_sum.end())
			ret = std::max(i - prefix_sum[sum], ret);
		else
			prefix_sum.insert({ sum, i });
	}
	return ret;
}

// TODO:
int Solution::numDupDigitsAtMostN(int n)
{
	auto numUniqueDigitsAtMostN = [](int n) -> int {
		auto num = std::to_string(n);
		int K = num.size();
		int ans = 0;
		int base = 10;
		int product = 1;
		for (int i = 0; i < K; i++) {
			ans += product;
			product *= base == 10 ? base - 1 : base;
			base--;
		}
		// first bit < num[0]
		product = num[0] - '0' - 1;
		base = 9;
		for (size_t i = 1; i < K; i++) {
			product *= base--;
		}
		if (product > 0)
			ans += product;
		// first bit == num[0]
		product = 1;
		for (size_t i = 1; i < K; i++) {
			product *= num[i] < num[0] ? num[i] - '0' - i + 1 :
							   num[i] - '0' - i;
		}
		if (product > 0)
			ans += product;
		return ans;
	};
	return n + 1 - numUniqueDigitsAtMostN(n);
}

int Solution::maxArea(int h, int w, std::vector<int> &horizontalCuts,
		      std::vector<int> &verticalCuts)
{
	horizontalCuts.push_back(0);
	verticalCuts.push_back(0);
	std::sort(horizontalCuts.begin(), horizontalCuts.end());
	std::sort(verticalCuts.begin(), verticalCuts.end());
	horizontalCuts.push_back(h);
	verticalCuts.push_back(w);
	int max_horizontal = 0;
	int max_vertical = 0;
	for (size_t i = 0; i + 1 < horizontalCuts.size(); i++)
		max_horizontal =
			std::max(max_horizontal,
				 horizontalCuts[i + 1] - horizontalCuts[i]);

	for (size_t i = 0; i + 1 < verticalCuts.size(); i++)
		max_vertical = std::max(max_vertical,
					verticalCuts[i + 1] - verticalCuts[i]);
	return max_horizontal * max_vertical;
}

Node *Solution::copyRandomList(Node *head)
{
	if (head == nullptr)
		return nullptr;
	Node *curr = head;
	while (curr != nullptr) {
		Node *new_node = new Node(curr->val);
		new_node->next = curr->next;
		curr->next = new_node;
		curr = new_node->next;
	}
	curr = head;
	while (curr != nullptr) {
		if (curr->random != nullptr)
			curr->next->random = curr->random->next;
		curr = curr->next->next;
	}
	curr = head;
	Node *ret = curr->next;
	while (curr != nullptr) {
		auto next = curr->next;
		curr->next = next->next;
		if (next->next != nullptr)
			next->next = next->next->next;
		curr = curr->next;
	}
	return ret;
}

ListNode *Solution::reverseKGroup(ListNode *head, int k)
{
	if (k == 1)
		return head;
	size_t length = 0;
	ListNode *curr = head;
	while (curr != nullptr & length <= k) {
		curr = curr->next;
		length++;
	}
	if (length < k)
		return head;
	ListNode dummy;
	dummy.next = head;
	ListNode *prev = &dummy;
	for (size_t i = 0; i < k; i++) {
		auto next = curr->next;
		curr->next = prev;
		prev = curr;
		curr = next;
	}
	return dummy.next;
}

std::vector<ListNode *> Solution::splitListToParts(ListNode *root, int k)
{
	std::vector<ListNode *> ret;
	ListNode *curr = root;
	int length = 0;
	while (curr != nullptr) {
		curr = curr->next;
		length++;
	}
	curr = root;
	int remainder = length % k;
	for (size_t i = 0; i < k; i++) {
		ret.push_back(curr);
		int width = length / k + (i < remainder ? 1 : 0);
		for (size_t j = 0; j + 1 < width; j++, curr = curr->next)
			;
		if (curr != nullptr) {
			ListNode *prev = curr;
			curr = curr->next;
			prev->next = nullptr;
		}
	}
	return ret;
}

std::vector<int> Solution::nextLargerNodes(ListNode *head)
{
	std::stack<std::pair<int, size_t> > stk;
	std::vector<int> ret(10000, 0);
	size_t i = 0;
	while (head) {
		while (!stk.empty() && stk.top().first < head->val) {
			ret[stk.top().second] = head->val;
			stk.pop();
		}
		stk.push(std::make_pair(head->val, i++));
		head = head->next;
	}
	ret.resize(i);
	return ret;
}

int Solution::maxNumberOfFamilies(int n,
				  std::vector<std::vector<int> > &reservedSeats)
{
	int ret = 0;
	std::unordered_map<int, int> occupied;
	constexpr int l = 0b00001111;
	constexpr int m = 0b00111100;
	constexpr int r = 0b11110000;
	for (const auto &seat : reservedSeats)
		if (1 < seat[1] && seat[1] < 10)
			occupied[seat[0]] |= (1 << (seat[1] - 2));
	for (auto [row, line] : occupied)
		if ((line & l) != 0 || (line & m) != 0 || (line & r) != 0)
			ret++;
	ret += (n - occupied.size()) << 1;
	return ret;
}
//TODO: dp
int Solution::tallestBillboard(std::vector<int> &rods)
{
	return 0;
}

int Solution::longestMountain(std::vector<int> &arr)
{
	int ret = 0;
	int l = 0;
	int r = 0;
	for (size_t i = 0; i + 1 < arr.size(); i++) {
		if (arr[i] < arr[i + 1]) {
			if (r != 0) {
				ret = std::max(l + r, ret);
				l = 0;
			}
			l++;
			r = 0;
		} else if (arr[i] == arr[i + 1]) {
			if (r != 0) {
				ret = std::max(l + r, ret);
				l = 0;
			}
			l = 0;
			r = 0;
		} else {
			r++;
		}
	}
	if (r != 0) {
		ret = std::max(l + r, ret);
		l = 0;
	}
	return ret;
}
int Solution::maxPerformance(int n, std::vector<int> &speed,
			     std::vector<int> &efficiency, int k)
{
	return 0;
}
std::vector<std::string> Solution::readBinaryWatch(int turnedOn)
{
	return std::vector<std::string>();
}
int Solution::numMatchingSubseq(std::string s, std::vector<std::string> &words)
{
	std::map<char, std::vector<int> > letters_map;
	for (int i = 0; i < s.length(); i++)
		letters_map[s[i]].push_back(i);
	int ret = 0;
	for (const auto &word : words) {
		int last_index = -1;
		bool isSubseq = true;
		for (auto letter : word) {
			if (letters_map.find(letter) == letters_map.end() ||
			    letters_map[letter].empty() ||
			    letters_map[letter].back() <= last_index) {
				isSubseq = false;
				break;
			}
			for (auto index : letters_map[letter])
				if (last_index < index) {
					last_index = index;
					break;
				}
		}
		ret += isSubseq ? 1 : 0;
	}
	return ret;
}
int Solution::maxSumTwoNoOverlap(std::vector<int> &nums, int firstLen,
				 int secondLen)
{
	int ret = 0;
	std::vector<int> prefix_sum = nums;
	for (size_t i = 0; i + 1 < prefix_sum.size(); i++)
		prefix_sum[i + 1] += prefix_sum[i];

	std::vector<int> a1(nums.size(), 0);
	a1[firstLen - 1] = prefix_sum[firstLen - 1];
	for (int i = firstLen - 1 + 1; i < nums.size(); i++)
		a1[i] = std::max(prefix_sum[i] - prefix_sum[i - firstLen],
				 a1[i - 1]);
	std::vector<int> a2(nums.size(), 0);
	for (int i = nums.size() - 1 - secondLen; i >= 0; i--)
		a2[i] = std::max(prefix_sum[i + secondLen] - prefix_sum[i],
				 a2[i + 1]);
	std::vector<int> a3(nums.size(), 0);
	std::vector<int> a4(nums.size(), 0);
	a3[secondLen - 1] = prefix_sum[secondLen - 1];
	for (int i = secondLen - 1 + 1; i < nums.size(); i++)
		a3[i] = std::max(prefix_sum[i] - prefix_sum[i - secondLen],
				 a3[i - 1]);
	for (int i = nums.size() - 1 - firstLen; i >= 0; i--)
		a4[i] = std::max(prefix_sum[i + firstLen] - prefix_sum[i],
				 a4[i + 1]);
	for (int i = 0; i < nums.size(); i++)
		ret = std::max(std::max(a1[i] + a2[i], a3[i] + a4[i]), ret);
	return ret;
}
int Solution::partitionDisjoint(std::vector<int> &nums)
{
	std::vector<int> suffix_min = nums;
	for (int i = suffix_min.size() - 1; i - 1 >= 0; i--)
		suffix_min[i - 1] = std::min(suffix_min[i], suffix_min[i - 1]);

	std::vector<int> prefix_max = nums;
	for (int i = 0; i + 1 < prefix_max.size(); i++)
		prefix_max[i + 1] = std::max(prefix_max[i], prefix_max[i + 1]);

	for (int i = 1; i < nums.size(); i++)
		if (prefix_max[i - 1] <= suffix_min[i])
			return i;
	return 0;
}
int Solution::findPaths(int m, int n, int maxMove, int startRow,
			int startColumn)
{
	constexpr int modulo = 1E9 + 7;
	if (maxMove == 0)
		return 0;
	std::vector<std::vector<std::vector<long> > > dp(
		maxMove,
		std::vector<std::vector<long> >(m, std::vector<long>(n, 0)));
	for (int i = 0; i < m; i++) {
		dp[0][i][0]++;
		dp[0][i][n - 1]++;
	}
	for (int i = 0; i < n; i++) {
		dp[0][0][i]++;
		dp[0][m - 1][i]++;
	}
	for (int i = 1; i < maxMove; i++) {
		for (int j = 0; j < m; j++)
			for (int k = 0; k < n; k++) {
				if (0 <= j - 1)
					dp[i][j][k] += dp[i - 1][j - 1][k];
				if (j + 1 < m)
					dp[i][j][k] += dp[i - 1][j + 1][k];
				if (0 <= k - 1)
					dp[i][j][k] += dp[i - 1][j][k - 1];
				if (k + 1 < n)
					dp[i][j][k] += dp[i - 1][j][k + 1];
				dp[i][j][k] %= modulo;
			}
	}
	long sum = 0;
	for (auto i = 0; i < maxMove; i++)
		sum += dp[i][startRow][startColumn];
	return sum % modulo;
}
// TODO:
double Solution::findMedianSortedArrays(std::vector<int> &nums1,
					std::vector<int> &nums2)
{
	if (nums1.empty())
		return (nums2[nums2.size() / 2] +
			nums2[(nums2.size() + 1) / 2]) /
		       2;
	if (nums2.empty())
		return (nums1[nums1.size() / 2] +
			nums1[(nums1.size() + 1) / 2]) /
		       2;
	int m = nums1.size(), n = nums2.size();

	return 0.0;
}
std::vector<int> Solution::countSmaller(std::vector<int> &nums)
{
	std::vector<int> ret(nums.size(), 0);
	std::vector<int> after;
	after.reserve(nums.size());
	after.push_back(nums[nums.size() - 1]);
	for (int i = nums.size() - 2; i >= 0; i--) {
		int lo = 0;
		int hi = after.size();
		while (lo < hi) {
			int mid = lo + (hi - lo) / 2;
			if (after[mid] < nums[i])
				lo = mid + 1;
			else
				hi = mid;
		}

		ret[i] = lo;
		after.insert(after.begin() + lo, nums[i]);
	}
	return ret;
}
std::vector<int> Solution::preorderTraversal(TreeNode *root)
{
	std::vector<int> ret;
	std::stack<TreeNode *> stk;
	stk.push(root);
	while (!stk.empty()) {
		auto curr = stk.top();
		stk.pop();
		if (curr != nullptr) {
			ret.push_back(curr->val);
			stk.push(curr->right);
			stk.push(curr->left);
		}
	}
	return ret;
}
bool Solution::isSymmetric(TreeNode *root)
{
	if (root == nullptr)
		return true;
	std::deque<TreeNode *> dq;
	dq.push_front(root->left);
	dq.push_back(root->right);
	while (!dq.empty()) {
		auto front = dq.front();
		dq.pop_front();
		auto back = dq.back();
		dq.pop_back();
		if (front == nullptr && back == nullptr)
			continue;
		if (front == nullptr || back == nullptr ||
		    front->val != back->val)
			return false;
		dq.push_front(front->right);
		dq.push_front(front->left);
		dq.push_back(back->left);
		dq.push_back(back->right);
	}
	return true;
}
bool Solution::isValidBST(TreeNode *root)
{
	if (root == nullptr)
		return true;
	std::stack<TreeNode *> stk;
	int last = INT_MIN;
	while (true) {
		while (root) {
			stk.push(root);
			root = root->left;
		}
		if (stk.empty())
			break;
		root = stk.top();
		stk.pop();
		if (root->val <= last)
			return false;
		last = root->val;
		root = root->right;
	}
	return true;
}
int Solution::snakesAndLadders(std::vector<std::vector<int> > &board)
{
	return 0;
}
int Solution::candy(std::vector<int> &ratings)
{
	std::vector<int> candies(ratings.size(), 1);
	int ret = ratings.size();
	int new_added = 0;
	do {
		new_added = 0;

		for (int i = 1; i < ratings.size(); i++) {
			if (ratings[i - 1] < ratings[i] &&
			    !(candies[i - 1] < candies[i])) {
				new_added += candies[i - 1] + 1 - candies[i];
				candies[i] = candies[i - 1] + 1;
			}
		}
		for (int i = ratings.size() - 1; i > 0; i--) {
			if (ratings[i] < ratings[i - 1] &&
			    !(candies[i] < candies[i - 1])) {
				new_added += candies[i] + 1 - candies[i - 1];
				candies[i - 1] = candies[i] + 1;
			}
		}
		ret += new_added;
	} while (new_added != 0);
	return ret;
}
int Solution::trailingZeroes(int n)
{
	int f = 5;
	int ret = 0;
	while (f < n) {
		ret += n / f;
		f *= 5;
	}
	return ret;
}
int Solution::preimageSizeFZF(int k)
{
	return 0;
}
std::string Solution::removeDuplicates(std::string s)
{
	std::stack<char> stk;
	for (auto chr : s)
		if (stk.empty() || stk.top() != chr)
			stk.push(chr);
		else
			stk.pop();
	std::string ret(stk.size(), ' ');
	for (int i = stk.size() - 1; i >= 0; i--) {
		ret[i] = stk.top();
		stk.pop();
	}
	return ret;
}

int Solution::longestOnes(std::vector<int> &nums, int k)
{
	int left = 0, right = 0, cnt = 0;
	while (right < nums.size()) {
		cnt += nums[right++] == 0 ? 1 : 0;
		if (cnt > k)
			cnt -= nums[left++] == 0 ? 1 : 0;
	}
	return right - left;
}
std::vector<int> Solution::grayCode(int n)
{
	size_t size = 1 << n;
	std::vector<int> ret(size, 0);
	for (size_t i = 1; i < size; i++)
		ret[i] = i ^ i << 1;
	return ret;
}
std::vector<int> Solution::findClosestElements(std::vector<int> &arr, int k,
					       int x)
{
	std::queue<int> q;
	std::vector<int> ret;
	for (auto n : arr)
		if (q.size() < k) {
			q.push(n);
		} else if (q.size() == k) {
			if (x - q.front() <= n - x)
				break;
			q.push(n);
			q.pop();
		}
	while (!q.empty()) {
		ret.push_back(q.front());
		q.pop();
	}
	return ret;
}
std::vector<std::vector<std::string> >
Solution::displayTable(std::vector<std::vector<std::string> > &orders)
{
	// orders[i]=[customerNamei,tableNumberi,foodItemi]
	std::sort(orders.begin(), orders.end(),
		  [](const std::vector<std::string> &lhs,
		     const std::vector<std::string> &rhs) {
			  if (std::stoi(lhs[1]) == std::stoi(rhs[1]))
				  return std::stoi(lhs[2]) < std::stoi(rhs[2]);
			  return std::stoi(lhs[1]) < std::stoi(rhs[1]);
		  });
	std::set<std::string> tables;
	std::map<std::string, int> food_number;
	int i = 0;
	for (const auto &order : orders) {
		tables.insert(order[1]);
		if (food_number.find(order[2]) == food_number.end())
			food_number.insert(std::make_pair(order[2], i++));
	}
	std::vector<std::vector<std::string> > ret(
		tables.size() + 1,
		std::vector<std::string>(food_number.size() + 1, "0"));

	ret[0][0] = "Table";
	for (const auto &[foodItem, index] : food_number)
		ret[0][index + 1] = foodItem;

	int row = 1;
	ret[row][0] = orders[0][1];
	auto lasttable = orders[0][1];
	for (const auto &order : orders) {
		if (order[1] != ret[row][0]) {
			row++;
			ret[row][0] = order[1];
		}
		ret[row][food_number[order[2]] + 1] = std::to_string(
			(std::stoi(ret[row][food_number[order[2]] + 1]) + 1));
	}
	//for (const auto &table : tables)

	return ret;
}
std::string Solution::customSortString(std::string order, std::string str)
{
	std::vector<int> hash_map(26, -1);
	for (int i = 0; i < order.size(); i++)
		hash_map[order[i] - 'a'] = i;
	std::sort(str.begin(), str.end(), [&](const char lhs, const char rhs) {
		return hash_map[lhs - 'a'] < hash_map[rhs - 'a'];
		// why can't be <=?
	});
	return str;
}
std::vector<std::vector<int> > Solution::fourSum(std::vector<int> &nums,
						 int target)
{
	std::vector<std::vector<int> > quadruplets;
	if (nums.size() < 4) {
		return quadruplets;
	}
	sort(nums.begin(), nums.end());
	int length = nums.size();
	for (int i = 0; i < length - 3; i++) {
		if (i > 0 && nums[i] == nums[i - 1]) {
			continue;
		}
		if (nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] >
		    target) {
			break;
		}
		if (nums[i] + nums[length - 3] + nums[length - 2] +
			    nums[length - 1] <
		    target) {
			continue;
		}
		for (int j = i + 1; j < length - 2; j++) {
			if (j > i + 1 && nums[j] == nums[j - 1]) {
				continue;
			}
			if (nums[i] + nums[j] + nums[j + 1] + nums[j + 2] >
			    target) {
				break;
			}
			if (nums[i] + nums[j] + nums[length - 2] +
				    nums[length - 1] <
			    target) {
				continue;
			}
			int left = j + 1, right = length - 1;
			while (left < right) {
				int sum = nums[i] + nums[j] + nums[left] +
					  nums[right];
				if (sum == target) {
					quadruplets.push_back(
						{ nums[i], nums[j], nums[left],
						  nums[right] });
					while (left < right &&
					       nums[left] == nums[left + 1]) {
						left++;
					}
					left++;
					while (left < right &&
					       nums[right] == nums[right - 1]) {
						right--;
					}
					right--;
				} else if (sum < target) {
					left++;
				} else {
					right--;
				}
			}
		}
	}
	return quadruplets;
}
std::vector<int> Solution::threeEqualParts(std::vector<int> &arr)
{
	std::vector<int> count = arr;
	for (int i = 0; i + 1 < arr.size(); i++)
		count[i + 1] += count[i];
	if (count.back() % 3 != 0)
		return std::vector<int>{ -1, -1 };
	if (count.back() == 0)
		return std::vector<int>{ 0, static_cast<int>(arr.size() - 1) };

	int suffix_zeros = 0;
	while (arr[arr.size() - 1 - suffix_zeros] == 0)
		suffix_zeros++;
	int i = 0;
	while (count[i] < count.back() / 3)
		i++;
	i += suffix_zeros;

	int j = i;
	while (count[j++] < count.back() / 3 * 2)
		;
	j += suffix_zeros;
	for (int k = 0; k <= j - 2 - i && k <= i && k <= arr.size() - 1 - j;
	     k++)
		if (arr[i - k] != arr[arr.size() - 1 - k] ||
		    arr[i - k] != arr[j - 1 - k])
			return std::vector<int>{ -1, -1 };
	return std::vector<int>{ i, j };
}
int Solution::findBestValue(std::vector<int> &arr, int target)
{
	return 0;
}
std::string Solution::pushDominoes(std::string dominoes)
{
	int last_R = -1;
	int last_L = -1;
	for (int i = 0; i < dominoes.length(); i++) {
		if (dominoes[i] == 'R') {
			last_R = i;
		} else if (dominoes[i] == 'L') {
			if (last_R <= last_L) {
				for (int j = last_L + 1; j < i; j++) {
					dominoes[j] == 'L';
				}
			} else {
				int left = last_R;
				int right = i;
				while (++left < --right) {
					dominoes[left] = 'R';
					dominoes[right] = 'L';
				}
			}
			last_L = i;
		}
	}
	if (last_L < last_R) {
		for (int i = dominoes.length() - 1; i >= 0; i--) {
			if (dominoes[i] == 'R')
				break;
			dominoes[i] = 'R';
		}
	}
	return dominoes;
}
int Solution::findIntegers(int n)
{
	std::vector<int> dp(32, 1);
	dp[1] = 2;
	for (size_t i = 0; i + 2 < dp.size(); i++)
		dp[i + 2] = dp[i] + dp[i + 1];

	if (n < 2)
		return dp[n];
	int nbits = 0;
	while (n >> nbits != 0) {
		++nbits;
	}

	if (n >> (nbits - 2) == 3) {
		return dp[n];
	} else {
		int mask = (1 << (nbits - 1)) - 1;
		return dp[n - 1] + findIntegers(n & mask);
	}
}

std::vector<int> Solution::beautifulArray(int n)
{
	return std::vector<int>();
}

std::vector<std::vector<int> >
Solution::updateMatrix(std::vector<std::vector<int> > &mat)
{
	auto mark = mat;
	int m = mat.size();
	int n = mat[0].size();
	auto dfs = make_y_combinator([&](auto &&dfs, int i, int j, int start_i,
					 int start_j) -> void {
		if (i < 0 || m <= i || j < 0 || n <= j || mark[i][j] == 0)
			return;
		if (mat[i][j] == 0) {
			mark[start_i][start_j] = 0;
			mat[start_i][start_j] =
				abs(start_i - i) + abs(start_j - j);
			return;
		}
		dfs(i, j - 1, start_i, start_j);
		dfs(i - 1, j, start_i, start_j);
		dfs(i, j + 1, start_i, start_j);
		dfs(i + 1, j, start_i, start_j);
	});
	for (int i = 0; i < mat.size(); i++)
		for (int j = 0; j < mat[0].size(); j++) {
			if (mat[i][j] == 0)
				continue;
			dfs(i, j, i, j);
		}
	return mat;
}

std::vector<std::vector<int> > Solution::levelOrder(Node *root)
{
	if (root == nullptr)
		return std::vector<std::vector<int> >();
	std::vector<std::vector<int> > ret;
	std::queue<Node *> q;
	q.push(root);
	int count = 0;
	while (!q.empty()) {
		count = q.size();
		std::vector<int> level;
		while (0 != count--) {
			Node *curr = q.front();
			q.pop();
			level.push_back(curr->val);
			for (auto child : curr->children)
				q.push(child);
		}
		ret.push_back(level);
	}
	Node *curr = q.front();
	return ret;
}

std::vector<int>
Solution::sumOfDistancesInTree(int n, std::vector<std::vector<int> > &edges)
{
	std::vector<std::unordered_set<int> > graph(n);
	for (auto &&edge : edges) {
		graph[edge[0]].insert(edge[1]);
		graph[edge[1]].insert(edge[0]);
	}

	std::vector<int> count(n, 1);
	std::vector<int> ret(n, 0);

	auto dfs = make_y_combinator(
		[&](auto &&dfs, int node = 0, int parent = -1) -> void {
			for (auto child : graph[node]) {
				if (child != parent) {
					dfs(child, node);
					count[node] += count[child];
					ret[node] += ret[child] + count[child];
				}
			}
		});

	auto dfs2 = make_y_combinator(
		[&](auto &&dfs2, int node = 0, int parent = -1) -> void {
			for (auto child : graph[node]) {
				if (child != parent) {
					ret[child] = ret[node] - count[child] +
						     n - count[child];
					dfs2(child, node);
				}
			}
		});
	dfs();
	dfs2();
	return ret;
}

int Solution::orderOfLargestPlusSign(int n,
				     std::vector<std::vector<int> > &mines)
{
	std::vector<std::vector<int> > grid(n, std::vector<int>(n, 1));
	for (auto &&mine : mines)
		grid[mine[0]][mine[1]] = 0;
	std::vector<std::vector<std::vector<int> > > dp(
		n, std::vector<std::vector<int> >(n, std::vector<int>(4, 0)));
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j + 1 < n; j++)
			dp[i][j + 1][0] = grid[i][j] == 0 ? 0 : dp[i][j][0] + 1;
		for (int j = n - 1; j - 1 >= 0; j--)
			dp[i][j - 1][1] = grid[i][j] == 0 ? 0 : dp[i][j][1] + 1;
	}
	for (size_t j = 0; j < n; j++) {
		for (size_t i = 0; i + 1 < n; i++)
			dp[i + 1][j][2] = grid[i][j] == 0 ? 0 : dp[i][j][2] + 1;
		for (int i = n - 1; i - 1 >= 0; i--)
			dp[i - 1][j][3] = grid[i][j] == 0 ? 0 : dp[i][j][3] + 1;
	}
	int ret = 0;
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			if (grid[i][j] == 1)
				ret = std::max(
					ret, *std::min_element(begin(dp[i][j]),
							       end(dp[i][j])));
		}
	}
	return 0;
}

// TODO: Dijkstra's algorithm 
int Solution::reachableNodes(std::vector<std::vector<int> > &edges,
			     int maxMoves, int n)
{
}
