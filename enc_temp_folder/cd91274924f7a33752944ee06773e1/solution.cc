#include "solution.h"
#include <queue>

#include "y_combinator.h"
#include <iostream>

int Solution::twoCitySchedCost(std::vector<std::vector<int>>& costs)
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
	std::multimap<unsigned int, std::vector<int>, std::greater<int>> mmap;
	for (auto pair : costs)
	{
		mmap.insert({ abs(pair[0] - pair[1]), pair });
	}
	int i = 0, j = 0;
	for (auto& e : mmap) {
		if ((count >> 1) <= i)
			res += e.second[1];
		else if ((count >> 1) <= j)
			res += e.second[0];
		else
			res += e.second[0] < e.second[1] ? (i++, e.second[0]) : (j++, e.second[1]);
	}
	//int ans = 118 + 259 + 54 + 667 + 184 + 577;
#endif
	return res;
}

std::vector<std::vector<int>> Solution::reconstructQueue(std::vector<std::vector<int>>& people)
{
	std::vector<std::vector<int>> res;
	int k = 0;
	std::sort(people.begin(), people.end(), [](std::vector<int> A, std::vector<int> B) { if (A.back() != B.back()) return A.back() < B.back(); return A.front() > B.front(); });
	size_t i = 0;
	while (i < people.size())
	{
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
	if (num < 0) return false;
	int count[2] = { 0 };
	while (num)
	{
		count[num & 0x1]++;
		num >>= 1;
	}
	return count[1] == 1 && count[0] % 2 == 0;
}

bool Solution::isSubsequence(std::string s, std::string t)
{
	if (t.length() < s.length()) return false;
	for (auto it = s.rbegin(); it != s.rend(); it++)
	{
		for (int i = t.length() - 1; i >= 0; i--)
		{
			if (t.at(i) == *it)
			{
				t.erase(i);
				break;
			}
			if (0 == i)
				return false;
		}
	}
	return true;
}

void Solution::sortColors(std::vector<int>& nums)
{
	int left = 0;
	int right = nums.size() - 1;
	int i = 0;
	while (left < right)
	{
		if (nums[i] == 0)
		{
			nums[left] ^= nums[i];
			nums[i] ^= nums[left];
			nums[left] ^= nums[i];
			left++;
		}
		else if (nums[i] == 2)
		{
			nums[right] ^= nums[i];
			nums[i] ^= nums[right];
			nums[right] ^= nums[i];
			right--;
		}
		i++;
	}
}

std::vector<int> Solution::largestDivisibleSubset(std::vector<int>& nums)
{
	std::vector<int> res;
	return res;
}

int Solution::findCheapestPrice(int n, std::vector<std::vector<int>>& flights, int src, int dst, int K)
{
	int res = 0;
	return res;
}

// TODO:
std::string Solution::validIPAddress(std::string IP)
{
	enum IPAddressType
	{
		IPv4,
		IPv6,
		Neither,
	};
	std::vector<std::string> strIPAddressType = { "IPv4", "IPv6", "Neither" };
	// IPV4
	if (7 <= IP.size() && IP.size() <= 15)
	{

		return strIPAddressType[IPAddressType::IPv4];
	}
	// IPV6
	if (15 <= IP.size() && IP.size() <= 39)
	{
		return strIPAddressType[IPAddressType::IPv6];
	}

	return strIPAddressType[IPAddressType::Neither];
}
void Solution::solve(std::vector<std::vector<char>>& board)
{
	auto dfs = make_y_combinator([](auto&& dfs, int row, int column, std::vector<std::vector<char>> grid) {
		if (row < 0 || row >= grid.size() || column < 0 || column >= grid[0].size() || grid[row][column] != 'O')
			return;
		grid[row][column] -= 9;
		dfs(row - 1, column, grid);
		dfs(row + 1, column, grid);
		dfs(row, column - 1, grid);
		dfs(row, column + 1, grid);
		});
	for (size_t i = 0; i < board.size(); i++)
		for (size_t j = 0; j < board[0].size(); j++)
			if (i == 0 || i == board.size() - 1 || j == 0 || j == board[0].size() - 1)
				if (board[i][j] == 'O')
					dfs(i, j, board);

	for (size_t i = 0; i < board.size(); i++)
		for (size_t j = 0; j < board[0].size(); j++)
			board[i][j] += 9;
}

int Solution::hIndex(std::vector<int>& citations)
{
	// sort
	std::sort(citations.begin(), citations.end(), std::greater<int>());

	// binary search
	int left = 0, right = citations.size();
	while (left < right)
	{
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
	for (size_t i = 1; i < n; i++)
	{
		count_of_last_permutation *= i;
	}
	while (!nums.empty())
	{
		int index = k / count_of_last_permutation;
		res.push_back(nums[index]);
		k %= count_of_last_permutation;
		count_of_last_permutation /= nums.size() - 1 ? nums.size() - 1 : 1;
		nums.erase(nums.begin() + index);
	}
	return res;
}

int Solution::calculateMinimumHP(std::vector<std::vector<int>>& dungeon)
{
	return 0;
}

// TODO:
int Solution::singleNumber(std::vector<int>& nums)
{
	int temps[2] = { 0 };
	int length = nums.size();
	for (size_t i = 0; i < length; i++)
	{
		if (1 == i % 2) {
			temps[0] ^= -nums[i];
			temps[1] ^= nums[i];
		}
		else {
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

std::vector<std::vector<int>> Solution::merge(std::vector<std::vector<int>>& intervals)
{
	//intervals = { {2,3},{4,5},{6,7},{8,9},{1,10} };
	std::vector<std::vector<int>> res;
	sort(intervals.begin(), intervals.end());
	res.push_back(intervals[0]);
	for (int i = 1; i < intervals.size(); i++)
	{
		if (intervals[i][0] <= res.back()[1])
		{
			res.back()[1] = res.back()[1] < intervals[i][1] ? intervals[i][1] : res.back()[1];
		}
		else
		{
			res.push_back(intervals[i]);
		}
	}
	return res;
}

int Solution::findDuplicate(std::vector<int>& nums)
{
	return 0;
}

// TODO : wrong
std::vector<std::string> Solution::findItinerary(std::vector<std::vector<std::string>>& tickets)
{
	return std::vector<std::string>();
}

int Solution::uniquePaths(int m, int n)
{
	std::vector<std::vector<int>> dp;
	dp.resize(m);
	for (auto& e : dp)
	{
		e.resize(n);
	}

	for (size_t i = 0; i < m; i++)
	{
		dp[i][0] = 1;
	}
	for (size_t i = 0; i < n; i++)
	{
		dp[0][i] = 1;
	}
	for (size_t i = 1; i < m; i++)
	{
		for (size_t j = 1; j < n; j++)
		{
			int raw = 0;
			while (raw <= i)
			{
				dp[i][j] += dp[raw][j - 1];
				raw++;
			}
		}
	}
	return dp[m - 1][n - 1];
}

int Solution::arrangeCoins(int n)
{
	long temp = static_cast<long>(n) * 8 + 1;
	return static_cast<int>((pow(temp, 0.5) - 1) / 2);
}

std::vector<int> Solution::prisonAfterNDays(std::vector<int>& cells, int N)
{
	if (0 == N)
		return cells;
	std::vector<std::vector<int>> results;
	results.push_back(cells);
	while (results.size() < 3 || results.back() != results[1])
	{
		std::vector<int> temp(8, 0);
		for (size_t i = 1; i < 7; i++)
			temp[i] = results.back()[i - 1] ^ results.back()[i + 1] ^ 1;
		results.push_back(temp);
	}
	results[0] = results[results.size() - 2];
	return results[N % (results.size() - 2)];
}

int Solution::hammingDistance(int x, int y)
{
	int n = x ^ y;
	int res = 0;
	while (n)
	{
		if (n & 1) res++;
		n >>= 1;
	}
	return res;
}

std::vector<int> Solution::plusOne(std::vector<int>& digits)
{
	size_t length = digits.size();
	std::vector<int> res(length, 0);
	for (int i = length - 1; i >= 0; i--)
	{
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
	while (0 == num % 2)
	{
		num /= 2;
	}
	while (0 == num % 3)
	{
		num /= 3;
	}
	while (0 == num % 5)
	{
		num /= 5;
	}
	return 1 == num;
}

int Solution::nthUglyNumber(int n)
{
	int i2 = 0, i3 = 0, i5 = 0;
	int dp[1690] = { 1 };
	for (size_t i = 1; i < n; i++)
	{
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

int Solution::islandPerimeter(std::vector<std::vector<int>>& grid)
{
	int res = 0;
	if (grid.empty() || grid[0].empty())
		return res;
	size_t m = grid.size(), n = grid[0].size();
	for (size_t i = 0; i < m; i++)
	{
		if (1 == grid[i][0])
			res++;
		if (1 == grid[i][n - 1])
			res++;
	}
	for (size_t i = 0; i < n; i++)
	{
		if (1 == grid[0][i])
			res++;
		if (1 == grid[m - 1][i])
			res++;
	}
	for (size_t i = 0; i < m - 1; i++)
	{
		for (size_t j = 0; j < n - 1; j++)
		{
			if (1 == grid[i][j] ^ grid[i][j + 1])
				res++;
			if (1 == grid[i][j] ^ grid[i + 1][j])
				res++;
		}
	}
	for (size_t i = 0; i < m - 1; i++)
	{
		if (1 == grid[i][n - 1] ^ grid[i + 1][n - 1])
			res++;
	}
	for (size_t i = 0; i < n - 1; i++)
	{
		if (1 == grid[m - 1][i] ^ grid[m - 1][i + 1])
			res++;
	}
	return res;
}

std::vector<std::vector<int>> Solution::threeSum(std::vector<int>& nums)
{
	//nums = { 0,0,0,0 };
	//nums = { -2,0,1,1,2 };
	std::vector<std::vector<int>> res;
	std::sort(nums.begin(), nums.end());

	for (size_t i = 0; i < nums.size(); i++)
	{
		if (0 < i && nums[i] == nums[i - 1])
			continue;
		if (0 < nums[i])
			break;
		size_t left = i + 1, right = nums.size() - 1;
		while (left < right)
		{
			if ((i + 1 < left && nums[left] == nums[left - 1]) || nums[left] + nums[right] < -nums[i]) {
				left++;
				continue;
			}
			if ((right < nums.size() - 1 && nums[right] == nums[right + 1]) || nums[left] + nums[right] > -nums[i]) {
				right--;
				continue;
			}
			if (nums[left] + nums[right] == -nums[i]) {
				res.push_back({ nums[i], nums[left++], nums[right] });
			}
		}
	}
	return res;
}

std::vector<std::vector<int>> Solution::subsets(std::vector<int>& nums)
{
	std::vector<std::vector<int>> res;
	size_t size = nums.size();
	std::vector<bool> bitmask(size + 1, false);
	while (false == bitmask.back())
	{
		std::vector<int> subset;
		for (size_t i = 0; i < size; i++)
		{
			if (bitmask[i])
				subset.push_back(nums[i]);
		}
		res.push_back(subset);
		size_t j = 0;
		bool carry = true;
		while (carry)
		{
			bitmask[j] = !bitmask[j];
			carry = !bitmask[j++];
		}
	}
	return res;
}

uint32_t Solution::reverseBits(uint32_t n)
{
	uint32_t res = 0;
	for (size_t i = 0; i < 32; i++)
	{
		res <<= 1;
		res |= (n & 1);
		n >>= 1;
	}
	return res;
}

double Solution::angleClock(int hour, int minutes)
{
	double minutes_angle = minutes * 6.0; /*360.0 / 60.0*/
	double hour_angle = (hour * 60.0 + minutes) * 0.5; /*360.0 / 12.0 / 60.0*/
	double res = 0.0 < minutes_angle - hour_angle ? minutes_angle - hour_angle : hour_angle - minutes_angle;
	return res < 180.0 ? res : 360.0 - res;
}

std::string Solution::reverseWords(std::string s)
{
	reverse(s.begin(), s.end());

	int n = s.size();
	int idx = 0;
	for (int start = 0; start < n; ++start) {
		if (s[start] != ' ') {
			if (idx != 0) s[idx++] = ' ';

			int end = start;
			while (end < n && s[end] != ' ') s[idx++] = s[end++];

			reverse(s.begin() + idx - (end - start), s.begin() + idx);

			start = end;
		}
	}
	s.erase(s.begin() + idx, s.end());
	return s;
}

char* Solution::reverseWords(char* s)
{
	return nullptr;
}

double Solution::myPow(double x, int n)
{
	if (0 == n) return 1.0;
	if (0.0 == x || 1.0 == x) return x;
	if (0 == n % 2 && 2 != n) return myPow(myPow(x, n >> 1), 2);
	if (n < 0) return myPow(1.0 / x, -n);
	return myPow(x, n - 1) * x;
}

int Solution::superPow(int a, std::vector<int>& b)
{
	return 0;
}

std::vector<int> Solution::topKFrequent(std::vector<int>& nums, int k)
{
	std::vector<int> res;
	std::unordered_map<int, int> hash;
	std::multimap<int, int, std::greater<int>> mmap;
	for (auto num : nums)
	{
		hash[num]++;
	}
	for (auto& e : hash)
	{
		mmap.insert({ e.second, e.first });
	}
	for (auto it = mmap.begin(); k--; it++)
	{
		res.push_back((*it).second);
	}
	return res;
}

std::string Solution::addBinary(std::string a, std::string b)
{
	a = "1111", b = "1111";
	std::string res;
	if (a.length() < b.length())
	{
		res = b;
		b = a;
		a = res;
	}
	else
	{
		res = a;
	}
	auto it = a.rbegin();
	int i = res.length() - 1;
	int carry = 0;
	for (auto jt = b.rbegin(); jt != b.rend(); it++, jt++, i--)
	{
		res[i] = *it + *jt - '0' + carry;
		carry = 0;
		if ('1' < res[i]) {
			res[i] -= 2;
			carry = 1;
		}
	}
	while (carry != 0 && it != a.rend())
	{
		res[i] = *it + carry;
		carry = 0;
		if ('1' < res[i]) {
			res[i] -= 2;
			carry = 1;
		}
		it++; i--;
	}
	return 1 == carry ? "1" + res : res;
}

bool Solution::canFinish(int numCourses, std::vector<std::vector<int>>& prerequisites)
{
	return false;
}

std::vector<int> Solution::findOrder(int numCourses, std::vector<std::vector<int>>& prerequisites)
{
	return std::vector<int>();
}

ListNode* Solution::reverseBetween(ListNode* head, int m, int n)
{
	ListNode* next = head->next;
	ListNode* prev = nullptr;
	ListNode* curr = head;
	ListNode* con;
	ListNode* tail;
	if (curr)
	{
		while (--m) {
			next = curr->next;
			prev = curr;
			curr = next;
			n--;
		}
		con = prev;
		tail = curr;
		while (n--)
		{
			next = curr->next;
			curr->next = prev;
			prev = curr;
			curr = next;
		}
		if (con)
			con->next = prev;
		else
			head = prev;
		tail->next = curr;
	}
	return head;
}

bool Solution::exist(std::vector<std::vector<char>>& board, std::string word)
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
		for (size_t i = 0; ; i++) {
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

char* Solution::convert(char* s, int numRows)
{
	return nullptr;
}

int Solution::numWaterBottles(int numBottles, int numExchange)
{
	return 0 == numBottles % (numExchange - 1) ? numBottles + numBottles / (numExchange - 1) - 1 : numBottles + numBottles / (numExchange - 1);
}

std::vector<int> Solution::singleNumbers(std::vector<int>& nums)
{
	return std::vector<int>();
}

std::vector<std::vector<int>> Solution::allPathsSourceTarget(std::vector<std::vector<int>>& graph)
{
	return std::vector<std::vector<int>>();
}

int Solution::findMin(std::vector<int>& nums)
{
	size_t left = 0;
	size_t right = nums.size() - 1;
	size_t idx = (left + right) >> 1;
	while (left < right)
	{
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

TreeNode* Solution::buildTree(std::vector<int>& inorder, std::vector<int>& postorder)
{
	if (inorder.empty())
		return nullptr;
	int llength = -1;
	while (inorder.at(llength++) != postorder.back())
		;
	std::vector<int> linorder(inorder.begin(), inorder.begin() + llength);
	std::vector<int> rinorder(inorder.begin() + llength + 1, inorder.end());
	std::vector<int> lpostorder(postorder.begin(), postorder.begin() + llength);
	std::vector<int> rpostorder(postorder.begin() + llength, postorder.end() - 1);
	TreeNode* root = new TreeNode(postorder.back());
	root->left = buildTree(linorder, lpostorder);
	root->right = buildTree(rinorder, rpostorder);
	return root;
}

int Solution::leastInterval(std::vector<char>& tasks, int n)
{
	if (0 == n) return tasks.size();
	std::unordered_map<char, int> hash;
	int res = 0;
	int a = 0, b = 0;
	for (auto& task : tasks)
		hash[task]++;
	for (auto& e : hash)
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

void Solution::merge(std::vector<int>& nums1, int m, std::vector<int>& nums2, int n)
{
	while (0 < m && 0 < n)
	{
		if (nums1[m - 1] < nums2[n - 1])
			nums1[m + n] = nums2[--n];
		else
			nums1[m + n] = nums1[--m];
	}
	while (0 < n)
	{
		nums1[n] = nums2[--n];
	}
}

std::vector<int> Solution::sortedSquares(std::vector<int>& A)
{
	std::vector<int> res(A.size());
	unsigned left = 0, right = A.size() - 1;
	for (int i = A.size() - 1; i >= 0; i--)
	{
		if (-A[left] < A[right])
		{
			res[i] = A[right] * A[right];
			right--;
		}
		else
		{
			res[i] = A[left] * A[left];
			left++;
		}
	}
	return res;
}

ListNode* Solution::mergeTwoLists(ListNode* l1, ListNode* l2)
{
#if recursion
	if (!l1)
		return l2;
	if (!l2)
		return l1;
	if (l1->val < l2->val)
	{
		l1->next = mergeTwoLists(l1->next, l2);
		return l1;
	}
	else
	{
		l2->next = mergeTwoLists(l1, l2->next);
		return l2;
	}
#else
	ListNode* sentinel = new ListNode();
	ListNode* res = sentinel;
	while (l1 && l2)
	{
		if (l1->val < l2->val)
		{
			res->next = l1;
			l1 = l1->next;
		}
		else
		{
			res->next = l2;
			l2 = l2->next;
		}
		res = res->next;
	}
	if (l1)
		res->next = l1;
	else
		res->next = l2;
	res = sentinel->next;
	delete sentinel;
	return res;
#endif // recursive
}

ListNode* Solution::mergeKLists(std::vector<ListNode*>& lists)
{
	// or use mergeTwoLists
	ListNode* sentinel = new ListNode();
	ListNode* res = sentinel;
	while (!lists.empty())
	{
		int min = INT_MAX;
		for (size_t i = 0; i < lists.size(); i++)
		{
			if (!lists[i]) {
				lists.erase(lists.begin() + i);
				i--;
			}
			else if (lists[i]->val < min) {
				min = lists[i]->val;
			}
		}
		for (size_t i = 0; i < lists.size(); i++)
		{
			if (lists[i] && lists[i]->val == min) {
				res->next = lists[i];
				lists[i] = lists[i]->next;
				break;
			}
		}
		res = res->next;
	}
	res = sentinel->next;
	delete sentinel;
	return res;
}

std::vector<std::string> Solution::wordBreak(std::string s, std::vector<std::string>& wordDict)
{
	std::vector<std::string> res;
	return res;
}

int Solution::integerBreak(int n)
{
	if (n < 4)
		return n - 1;
	int res = 1;
	while (4 < n)
	{
		n -= 3;
		res *= 3;
	}
	return res * n;
}

int Solution::climbStairs(int n)
{
	std::vector<int> dp(n + 1, 1);
	for (size_t i = 2; i < n + 1; i++)
	{
		dp[i] = dp[i - 1] + dp[i - 2];
	}
	return dp[n];
}

std::string Solution::getHint(std::string secret, std::string guess)
{
	int bulls_count = 0, cows_count = 0;
	for (int i = secret.length() - 1; i >= 0; i--)
	{
		if (secret[i] == guess[i])
		{
			bulls_count++;
			secret.erase(secret.begin() + i);
			guess.erase(guess.begin() + i);
		}
	}
	std::unordered_map<char, int> map_secret, map_guess;
	for (auto& chr : secret)
		map_secret[chr]++;
	for (auto& chr : guess)
		map_guess[chr]++;
	for (auto& e : map_secret)
		if (map_guess[e.first])
			cows_count += e.second < map_guess[e.first] ? e.second : map_guess[e.first];
	std::string hint;
	return std::to_string(bulls_count) + "A" + std::to_string(cows_count) + "B";
}

bool Solution::detectCapitalUse(std::string word)
{
	if ('Z' < word[0])
	{
		for (auto& letter : word)
			if (letter <= 'Z')
				return false;
	}
	else
	{
		for (size_t i = 1; i < word.length() - 1; i++)
		{
			if ((word[i] - '[') * (word[i + 1] - '[') < 0)
				return false;
		}
	}
	return true;
}

std::vector<int> Solution::smallestRange(std::vector<std::vector<int>>& nums)
{
	return std::vector<int>();
}

int Solution::countGoodTriplets(std::vector<int>& arr, int a, int b, int c)
{
	//std::sort(arr.begin(), arr.end());
	int count = 0;
	for (size_t i = 0; i < arr.size() - 2; i++)
		for (size_t j = i + 1; j < arr.size() - 1; j++)
			for (size_t k = j + 1; k < arr.size(); k++)
				if ((arr[i] - arr[j] <= a) && (-a <= arr[i] - arr[j]) && (arr[j] - arr[k] <= b) && (-b <= arr[j] - arr[k]) && (arr[i] - arr[k] <= c) && (-c <= arr[i] - arr[k]))
					count++;
	return count;
}

int Solution::getWinner(std::vector<int>& arr, int k)
{
	int winner = arr[0];
	int win_count = 0;
	for (size_t i = 1; i < arr.size(); i++)
	{
		if (arr[i] < winner)
		{
			win_count++;
		}
		else
		{
			win_count = 1;
			winner = arr[i];
		}
		if (win_count == k)
			return winner;
	}
	return winner;
}

int Solution::minSwaps(std::vector<std::vector<int>>& grid)
{
	int n = grid.size();
	int m = grid[0].size();

	if (n == 1)
		return 0;
	if (n == 2)
	{
		if (grid[0].back() == 0)
			return 0;
		if (grid[1].back() == 0)
			return 1;
		return -1;
	}
	int res = 0;
	size_t i = 0;
	bool flag = true;
	for (i = 0; i < n; i++)
	{
		flag = true;
		for (int j = m - 1; j >= m - n + 1; j--)
		{
			if (1 == grid[i][j])
			{
				flag = false;
				break;
			}
		}
		if (flag)
		{
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

int Solution::maxSum(std::vector<int>& nums1, std::vector<int>& nums2)
{
	std::unordered_map<int, bool> umap1, umap2;
	for (auto& num : nums1)
	{
		umap1[num] = true;
	}
	for (auto& num : nums2)
	{
		umap2[num] = true;
	}
	std::vector<int> nums_common;
	for (auto& e : umap1)
	{
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
	std::vector<std::vector<int>> dp(nums.size() - 1, { 0, 0 });
	std::vector<int> dp2(nums.size() - 1, 0);
	dp[0][0] = nums[0];
	dp[1][0] = std::max(nums[0], nums[1]);
	dp[0][1] = nums[1];
	dp[1][1] = std::max(nums[1], nums[2]);
	for (size_t i = 2; i < nums.size() - 1; i++)
	{
		dp[i][0] = std::max(dp[i - 2][0] + nums[i], dp[i - 1][0]);
		dp[i][1] = std::max(dp[i - 2][1] + nums[i + 1], dp[i - 1][1]);
	}
	return std::max(dp.back().at(0), dp.back().back());
#endif
}
int Solution::rob(TreeNode* root)
{
	static std::unordered_map<TreeNode*, int> robmap;
	if (!root) return 0;
	if (robmap[root]) return robmap[root];
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
int Solution::diameterOfBinaryTree(TreeNode* root)
{
	int ans = 0;
	static auto height = make_y_combinator([&](auto&& height, TreeNode* node) -> int {
		if (node == nullptr) return -1;
		int L = height(node->left);
		int R = height(node->right);
		ans = std::max(ans, L + 1 + R + 1);
		return std::max(L, R) + 1;
		});
	height(root);
	return ans;
}

std::vector<int> Solution::findDuplicates(std::vector<int>& nums)
{
	std::vector<int> res;
	for (size_t i = 0; i < nums.size(); i++)
	{
		size_t index = 0 < nums[i] ? nums[i] - 1 : -nums[i] - 1;
		if (nums[index] < 0)
			res.push_back(index + 1);
		nums[index] *= -1;
	}
	return res;
}

std::vector<int> Solution::findDisappearedNumbers(std::vector<int>& nums)
{
	std::vector<int> res;
	for (size_t i = 0; i < nums.size(); i++)
	{
		size_t index = 0 < nums[i] ? nums[i] - 1 : -nums[i] - 1;
		if (nums[index] > 0)
			nums[index] *= -1;
	}
	for (size_t i = 0; i < nums.size(); i++)
	{
		if (0 < nums[i])
			res.push_back(i + 1);
	}
	return res;
}

std::vector<std::vector<int>> Solution::verticalOrder(TreeNode* root)
{
	std::vector<std::vector<int>> res;
	std::queue<std::pair<TreeNode*, int>> q;
	std::map<int, std::vector<int>> map;
	q.push(std::make_pair(root, 0));
	while (!q.empty())
	{
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
	for (auto e : map)
	{
		res.push_back(e.second);
	}
	return res;
}

std::vector<std::vector<int>> Solution::verticalTraversal(TreeNode* root)
{
	std::vector<std::vector<int>> res;
	std::queue<std::pair<TreeNode*, std::pair<int, int>>> q;
	std::map<int, std::vector<std::pair<int, int>>> map;
	q.push(std::make_pair(root, std::make_pair(0, 0)));
	while (!q.empty())
	{
		auto node = q.front().first;
		auto x = q.front().second.first;
		auto y = q.front().second.second;
		auto val = node->val;
		if (0 == map.count(x))
			map[x] = std::vector<std::pair<int, int>>();
		map[x].push_back(std::make_pair(node->val, y));
		q.pop();
		if (node->left)
			q.push(std::make_pair(node->left, std::make_pair(x - 1, y + 1)));
		if (node->right)
			q.push(std::make_pair(node->right, std::make_pair(x + 1, y + 1)));
	}
	for (auto& e : map)
		std::sort(e.second.begin(), e.second.end(), [](std::pair<int, int> A, std::pair<int, int> B) {
		if (A.second != B.second) return A.second < B.second; return A.first < B.first;
			});
	for (auto& e : map)
	{
		std::vector<int> temp;
		for (auto& pair : e.second)
			temp.push_back(pair.first);
		res.push_back(temp);
	}
	return res;
}

std::vector<int> Solution::inorderTraversal(TreeNode* root)
{
	std::vector<int> res;
	TreeNode* predecessor = nullptr;
	TreeNode* curr = root;
	while (curr)
	{
		if (!curr->left)
		{
			res.push_back(curr->val);
			curr = curr->right;
		}
		else
		{
			predecessor = curr->left;
			while (predecessor->right && predecessor->right != curr)
				predecessor = predecessor->right;
			if (!predecessor->right)
			{
				predecessor->right = curr;
				//res.push_back(curr->val); // caution!!!
				curr = curr->left;
			}
			else
			{
				predecessor->right = nullptr;
				res.push_back(curr->val);
				curr = curr->right;
			}
		}
	}
	return res;
}

void Solution::recoverTree(TreeNode* root)
{
	TreeNode* x = nullptr;
	TreeNode* y = nullptr;
	TreeNode* prev = nullptr;
	TreeNode* curr = root;
	std::stack<TreeNode*> s;
	while (true)
	{
		while (curr)
		{
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

bool Solution::hasPathSum(TreeNode* root, int sum)
{
	bool left_has_path_sum = false, right_has_path_sum = false, curr = false;
	if (root)
	{
		if (!root->left && !root->right) curr = (sum == root->val);
		if (root->left)
			left_has_path_sum = hasPathSum(root->left, sum - root->val);
		if (root->right)
			right_has_path_sum = hasPathSum(root->right, sum - root->val);
	}
	return curr || left_has_path_sum || right_has_path_sum;
}
#if ROOT_TO_LEAF
std::vector<std::vector<int>> Solution::pathSum(TreeNode* root, int sum)
{
	std::vector<std::vector<int>> res;
	return res;
}
#else
// TODO:
int Solution::pathSum(TreeNode* root, int sum)
{
	static int first = sum;
	int left_path_sum = 0, right_path_sum = 0, curr = 0;
	if (root) {
		curr = sum == root->val ? 1 : 0;
		if (sum == first) {
			if (root->left)
				left_path_sum = pathSum(root->left, sum - root->val) + pathSum(root->left, sum);
			if (root->right)
				right_path_sum = pathSum(root->right, sum - root->val) + pathSum(root->right, sum);
		}
		else {
			if (root->left)
				left_path_sum = pathSum(root->left, sum - root->val);
			if (root->right)
				right_path_sum = pathSum(root->right, sum - root->val);
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

int Solution::closestValue(TreeNode* root, double target)
{
	return 0;
}

int Solution::orangesRotting(std::vector<std::vector<int>>& grid)
{
	std::queue<std::pair<size_t, size_t>> q;
	int fresh = 0;
	for (size_t row = 0; row < grid.size(); row++)
		for (size_t column = 0; column < grid[0].size(); column++)
			if (grid[row][column] == 2)
				q.push(std::make_pair(row, column));
			else if (grid[row][column] == 1)
				fresh++;
	unsigned size = 0;
	int rot_times = 0;
	while (!q.empty() && fresh != 0)
	{
		rot_times++;
		size = q.size();
		while (size-- != 0)
		{
			auto rotted_coord = q.front();
			q.pop();
			std::vector<std::pair<size_t, size_t>> rotting_coords;
			if (0 < rotted_coord.first) rotting_coords.push_back(std::make_pair(rotted_coord.first - 1, rotted_coord.second));
			if (rotted_coord.first + 1 < grid.size()) rotting_coords.push_back(std::make_pair(rotted_coord.first + 1, rotted_coord.second));
			if (0 < rotted_coord.second) rotting_coords.push_back(std::make_pair(rotted_coord.first, rotted_coord.second - 1));
			if (rotted_coord.second + 1 < grid[0].size()) rotting_coords.push_back(std::make_pair(rotted_coord.first, rotted_coord.second + 1));
			for (auto coord : rotting_coords)
				if (1 == grid[coord.first][coord.second]) {
					fresh--;
					grid[coord.first][coord.second] = 2;
					q.push(std::make_pair(coord.first, coord.second));
				}
		}
	}
	return fresh == 0 ? rot_times : -1;
}

int Solution::titleToNumber(std::string s)
{
	int res = 0;
	for (const auto& chr : s)
	{
		res *= 26;
		res += chr - 'A';
	}
	return res;
}

int Solution::countBinarySubstrings(std::string s)
{
	return 0;
}

std::string Solution::makeGood(std::string s)
{
	bool is_great = false;
	while (!is_great)
	{
		is_great = true;
		for (int i = s.length() - 2; i >= 0; i--)
			if (tolower(s[i]) == tolower(s[i + 1]) && s[i] != s[i + 1])
			{
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
int Solution::maxNonOverlapping(std::vector<int>& nums, int target)
{
	std::map<int, int> map;
	for (size_t i = 0; i < nums.size(); i++)
	{

	}
	return 0;
}

std::vector<std::vector<int>> Solution::generate(int numRows)
{
	std::vector<std::vector<int>> res;
	for (size_t i = 0; i < numRows; i++)
	{
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
	while (lo < hi)
	{
		long temp = res[i++] * hi--;
		res[i] = temp / lo++;
		res[rowIndex - i - 1] = res[i];
	}
	return res;
}

Node* Solution::cloneGraph(Node* node)
{
	if (!node)
		return nullptr;
	static std::map<Node*, Node*> map;
	std::vector<Node*> neighbors;
	Node* res = new Node(node->val);
	map[node] = res;
	for (auto& neighbor : node->neighbors)
	{
		if (map.count(neighbor) == 0)
			map[neighbor] = cloneGraph(neighbor);
		neighbors.push_back(map[neighbor]);
	}
	res->neighbors = neighbors;
	return res;
}

ListNode* Solution::addTwoNumbers(ListNode* l1, ListNode* l2)
{
	int sum = 0;
	ListNode* l3 = nullptr;
	ListNode** node = &l3;
	while (l1 != nullptr || l2 != nullptr || 0 < sum)
	{
		if (l1 != nullptr)
		{
			sum += l1->val;
			l1 = l1->next;
		}
		if (l2 != nullptr)
		{
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
	while (0 != carry || it != num1.rend() || jt != num2.rend())
	{
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
	for (size_t i = 0; i < num1.length(); i++)
	{

	}
	return std::string();
}

bool Solution::isValid(std::string s)
{
	int n = s.size();
	if ((n & n - 1) != 0)
		return false;

	std::unordered_map<char, char> pairs = {
		{')', '('},
		{']', '['},
		{'}', '{'}
	};
	std::stack<char> stk;
	for (char ch : s) {
		if (pairs.count(ch)) {
			if (stk.empty() || stk.top() != pairs[ch]) {
				return false;
			}
			stk.pop();
		}
		else {
			stk.push(ch);
		}
	}
	return stk.empty();
}

int Solution::longestPalindrome(std::string s)
{
	std::unordered_map<char, int> umap;
	for (const auto& ch : s)
		umap[ch]++;
	int res = 0;
	for (const auto& e : umap) {
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
	std::unordered_map<char, bool> umap(false);
	for (const auto& ch : s)
		umap[ch] = !umap[ch];
	int count = 0;
	for (const auto& e : umap)
		if (e.second)
			count++;
	return count < 2;
}

int Solution::removeBoxes(std::vector<int>& boxes)
{
	return 0;
}

int Solution::eraseOverlapIntervals(std::vector<std::vector<int>>& intervals)
{
	std::sort(intervals.begin(), intervals.end());
	return 0;
}

std::vector<int> Solution::findPermutation(std::string s)
{
	return std::vector<int>();
}

bool Solution::threeConsecutiveOdds(std::vector<int>& arr)
{
	for (size_t i = 0; i + 2 < arr.size(); i++)
	{
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

int Solution::maxDistance(std::vector<int>& position, int m)
{
	std::sort(position.begin(), position.end());
	std::vector<int> distance;
	for (size_t i = 0; i + 1 < position.size(); i++)
		distance.push_back(position.at(i + 1) - position.at(i));
	if (2 == m)
		return position.back() - position.at(0);
	if (3 == m)
		return std::min(position.at(position.size() - 2) - position.at(0), position.back() - position.at(1));
	return 0;
}

int Solution::minDays(int n)
{
	static std::unordered_map<int, int> dp;
	if (n <= 1)
		return n;
	if (dp.count(n) == 0)
		dp[n] = 1 + std::min(n % 2 + minDays(n / 2), n % 3 + minDays(n / 3));
	return dp[n];
}

int Solution::maxProfit_1st(std::vector<int>& prices)
{
	int min_price = INT_MAX;
	int max_profit = 0;
	for (auto& price : prices)
	{
		if (price < min_price)
			min_price = price;
		if (max_profit < price - min_price)
			max_profit = price - min_price;
	}
	return max_profit;
}

int Solution::maxProfit_2nd(std::vector<int>& prices)
{
	return 0;
}

// TODO:
int Solution::maxProfit_3rd(std::vector<int>& prices)
{
	std::vector<int> profits;
	auto profit_once = maxProfit_1st(prices);
	if (0 == profit_once)
		return 0;
	for (size_t i = 1; i + 1 < prices.size(); i++)
	{
		// max = maxbefore + max_post
		std::vector<int> prices_before(prices.begin(), prices.begin() + i + 1);
		std::vector<int> prices_after(prices.begin() + i + 1, prices.end());
		// max before
		//maxProfit_1st(prices_before);
		// max after
		//maxProfit_1st(prices_after);
		profits.push_back(maxProfit_1st(prices_before) + maxProfit_1st(prices_after));
		// sort max before and after
		//add ttwo back()
	}
	int max_profit = 0;
	for (const auto& profit : profits)
		if (max_profit < profit)
			max_profit = profit;
	if (max_profit < profit_once)
		max_profit = profit_once;
	return max_profit;
}

int Solution::maxProfit_4th(std::vector<int>& prices)
{
	return 0;
}

int Solution::maxProfit_5th(std::vector<int>& prices)
{
	return 0;
}

std::vector<int> Solution::distributeCandies(int candies, int num_people)
{
	candies = 44;
	int row_count = 0;
	int square_of_num_people = num_people * num_people;
	int limit = 0;
	while (limit < candies)
	{
		row_count++;
		limit = row_count * num_people * (num_people + 1) / 2 + square_of_num_people * (row_count - 1) * row_count / 2;
	}
	int minus = limit - candies;
	std::vector<int> res(num_people, 0);
	size_t i = 0;
	for (; i < num_people; i++)
	{
		res[i] = row_count * (i + 1) + num_people * (row_count - 1) * row_count / 2;
	}
	while (0 < minus)
	{
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
			word_list.push_back(S.substr(last_index, i - last_index));
			last_index = i + 1;
		}
	word_list.push_back(S.substr(last_index, S.length() - last_index));
	std::string res;
	std::string suffix = "maa";
	auto isVowel = [](char ch) { return ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u'
		|| ch == 'A' || ch == 'E' || ch == 'I' || ch == 'O' || ch == 'U'; };
	for (size_t i = 0; i < word_list.size(); i++)
	{
		if (isVowel(word_list.at(i).at(0))) {
			res.append(word_list.at(i));
		}
		else {
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

void Solution::reorderList(ListNode* head)
{
	ListNode* fast = head, * slow = head;
	while (fast && fast->next)
	{
		slow = slow->next;
		fast = fast->next->next;
	}

	//reverse second half
	ListNode* prev = nullptr;
	ListNode* curr = slow;
	while (curr)
	{
		ListNode* next = curr->next;
		curr->next = prev;
		prev = curr;
		curr = next;
	}
	slow = prev;

	// merge
	while (slow && slow->next)
	{
		auto temp = head->next;
		head->next = slow;
		head = temp;
		temp = slow->next;
		slow->next = head;
		slow = temp;
	}
}

std::vector<std::vector<char>> Solution::updateBoard(std::vector<std::vector<char>>& board, std::vector<int>& click)
{
	int dir_x[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int dir_y[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

	auto dfs = make_y_combinator([&](auto&& dfs, int row, int column) {
		if (row < 0 || row >= board.size() || column < 0 || column >= board[0].size() || board[row][column] != 'E')
			return;
		else if ('E' == board[row][column]) {
			int count = 0;
			int row_size = board.size(), column_size = board.at(0).size();
			for (size_t i = 0; i < 8; i++)
			{
				auto x = row + dir_x[i];
				auto y = column + dir_y[i];
				if (0 <= x && x < row_size && 0 <= y && y < column_size && 'M' == board[x][y])
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
			}
			else
			{
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

int Solution::minDepth(TreeNode* root)
{
	if (root == nullptr) return 0;
	std::queue<TreeNode*> queue;
	queue.push(root);
	int res = 0;
	int count = 0;
	while (!queue.empty())
	{
		res++;
		count = queue.size();
		while (0 != count--)
		{
			TreeNode* curr = queue.front();
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

int Solution::maxDepth(TreeNode* root)
{
	if (root == nullptr) return 0;
	std::queue<TreeNode*> queue;
	queue.push(root);
	int res = 0;
	int count = 0;
	while (!queue.empty())
	{
		res++;
		count = queue.size();
		while (0 != count--)
		{
			TreeNode* curr = queue.front();
			queue.pop();
			if (curr->left)
				queue.push(curr->left);
			if (curr->right)
				queue.push(curr->right);
		}
	}
	return res;
}

std::vector<int> Solution::sortArrayByParity(std::vector<int>& A)
{
	int left = 0, right = A.size() - 1;
	while (left < right)
	{
		if (0 == A[left] % 2)
		{
			left++;
			continue;
		}
		if (1 == A[right] % 2)
		{
			right--;
			continue;
		}
		std::swap(A[left++], A[right--]);
	}
	return A;
}

bool Solution::isBalanced(TreeNode* root)
{
	auto helper = make_y_combinator([](auto&& helper, TreeNode* root)->int {
		if (root == nullptr)
			return 0;
		int leftHeight = 0, rightHeight = 0;
		if (-1 == (leftHeight = helper(root->left)) || -1 == (rightHeight = helper(root->right)) || 1 < abs(leftHeight - rightHeight))
			return -1;
		else
			return std::max(leftHeight, rightHeight) + 1;
		return 0;
		});
	return 0 <= helper(root);
	return false;
}
int Solution::numOfMinutes(int n, int headID, std::vector<int>& manager, std::vector<int>& informTime)
{
	auto helper = make_y_combinator([](auto&& helper, Node* node) {
		if (node->neighbors.empty())
			return node->val;
		auto max_time = 0;
		for (size_t i = 0; i < node->neighbors.size(); i++)
			max_time = std::max(max_time, helper(node->neighbors[i]) + node->val);
		return max_time;
		});
	std::vector<Node> employees;
	for (size_t i = 0; i < n; i++)
	{
		Node employee(informTime[i]);
		employees.push_back(employee);
	}
	for (size_t i = 0; i < n; i++)
		if (headID != i)
			employees[manager[i]].neighbors.push_back(&employees[i]);
	int res;
	res = helper(&employees[headID]);
	return res;
}

bool Solution::isCousins(TreeNode* root, int x, int y)
{
	if (root == nullptr) return 0;
	std::queue<TreeNode*> queue;
	queue.push(root);
	int count = 0;
	std::map<int, int> parent;
	std::vector<std::vector<int>> level_order;
	while (!queue.empty())
	{
		count = queue.size();
		std::vector<int> level;
		while (0 != count--)
		{
			TreeNode* curr = queue.front();
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

bool Solution::judgePoint24(std::vector<int>& nums)
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
	while (++offset < (s.length() >> 1) + 1)
	{
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

int Solution::sumOfLeftLeaves(TreeNode* root)
{
	int res = 0;
	if (root == nullptr)
		return 0;
	if (root->left != nullptr && root->left->left == nullptr && root->left->left == nullptr)
		res += root->left->val;
	return res + sumOfLeftLeaves(root->right);
}

// TODO:
std::vector<std::vector<int>> Solution::findSubsequences(std::vector<int>& nums)
{
	int upperbound = 1 << nums.size();
	std::vector<std::vector<int>> result;
	for (size_t bitmask = 0; bitmask < upperbound; bitmask++)
	{
		if (0 == (bitmask & (bitmask - 1))) //only one element
			continue;
		std::vector<int> subsequence;
		for (int idx = nums.size() - 1; 0 <= idx; idx--)
		{
			if (1 == ((bitmask >> idx) & 1))
				subsequence.push_back(nums.at(nums.size() - 1 - idx));
		}
		bool flag = true;
		for (size_t i = 0; i + 1 < subsequence.size(); i++)
			if (subsequence[i + 1] < subsequence[i])
			{
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

int Solution::mincostTickets(std::vector<int>& days, std::vector<int>& costs)
{
	std::vector<int> dp(days.size(), 0);
	dp[0] = std::min(std::min(costs[0], costs[1]), costs[2]);
	for (size_t i = 1; i < days.size(); i++)
	{
		std::vector<int> prices(3, dp[i - 1]);
		// use  1-day pass
		prices[0] = dp[i - 1] + costs[0];
		// use  7-day pass
		if (days[i] - days[0] < 7)
			prices[1] = costs[1];
		else
		{
			int last = i;
			while (days[i] - 7 < days[last])
				last--;
			prices[1] = costs[1] + dp[last];
		}
		// use  30-day pass
		if (days[i] - days[0] < 30)
			prices[2] = costs[2];
		else
		{
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
	for (size_t i = 1; i <= n; i++)
	{
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

ListNode* Solution::sortList(ListNode* head)
{
	// recursion
	if (head == nullptr || head->next == nullptr)
		return head;
	ListNode* slow = head;
	ListNode* fast = head->next;
	while (fast && fast->next) {
		slow = slow->next;
		fast = fast->next->next;
	}
	ListNode* second_half = sortList(slow->next);
	slow->next = nullptr;
	ListNode* first_half = sortList(head);
	return mergeTwoLists(first_half, second_half);
	// TODO: iteration
}

std::vector<int> Solution::countBits(int num)
{
	std::vector<int> dp(num + 1, 0);
	size_t last = 0;
	for (size_t i = 1; i <= num; i++)
	{
		if ((i & (i - 1)) == 0) {
			dp[i] = 1;
			last = i;
		}
		else {
			dp[i] = dp[last] + dp[i - last];
		}
	}
	return dp;
}

TreeNode* Solution::mergeTrees(TreeNode* t1, TreeNode* t2)
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
std::vector<int> Solution::findRightInterval(std::vector<std::vector<int>>& intervals)
{
	std::vector<int> res;
	std::map<std::pair<int, int>, int> umap;
	int idx = 0;
	for (const auto& interval : intervals)
		umap[std::make_pair(interval[0], interval[1])] = idx++;
	std::vector<std::vector<int>> sorted_intervals = intervals;
	std::sort(sorted_intervals.begin(), sorted_intervals.end());
	for (const auto& interval : intervals)
	{
		auto left = 0;
		auto right = intervals.size();
		if (sorted_intervals.back()[0] < interval[1])
			res.push_back(-1);
		else
		{
			while (left < right)
			{
				auto mid = left + (right - left) / 2;
				if (sorted_intervals[mid][0] - interval[1] < 0) // <=
					left = mid + 1;
				else
					right = mid;
			}
			res.push_back(umap[std::make_pair(sorted_intervals[left][0], sorted_intervals[left][1])]);
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
	std::vector<int> values = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
	std::vector<std::string> symbols = { "M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I" };

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
	std::map<char, int> m = {
		{'I', 1},
		{'V', 5},
		{'X', 10},
		{'L', 50},
		{'C', 100},
		{'D', 500},
		{'M', 1000}
	};
	for (size_t i = 0; i + 1 < s.length(); i++)
	{
		if (m[s[i]] < m[s[i + 1]])
			res -= m[s[i]];
		else
			res += m[s[i]];
	}
	res += m[s.back()];
	return res;
}

bool Solution::judgeCircle(std::string moves) {
	int deltaX = 0, deltaY = 0;
	for (const auto& chr : moves)
	{
		switch (chr) {
		case 'R':deltaX++; break;
		case 'L':deltaX--; break;
		case 'U':deltaY++; break;
		case 'D':deltaY--; break;
		}
	}
	return deltaX == 0 && deltaY == 0;
}

int Solution::rand10()
{
	auto rand7 = []() { return rand() % 7 + 1; };
	int a = 7, b = 7;

	while (a == 7) a = rand7();
	while (b > 5) b = rand7();

	return a & 1 ? b : b + 5;
}

bool Solution::hasCycle(ListNode* head)
{
	ListNode* fast = head;
	ListNode* slow = head;
	while (fast != nullptr && fast->next != nullptr)
	{
		fast = fast->next->next;
		slow = slow->next;
		if (fast == slow)
			return true;
	}
	return false;
}

ListNode* Solution::getIntersectionNode(ListNode* headA, ListNode* headB)
{
	ListNode* pA = headA;
	ListNode* pB = headB;
	while (pA != pB)
	{
		pA = pA != nullptr ? pA->next : headB;
		pB = pB != nullptr ? pB->next : headA;
	}
	return pA;
}

std::vector<std::string> Solution::findRestaurant(std::vector<std::string>& list1, std::vector<std::string>& list2)
{
	return std::vector<std::string>();
}

std::string Solution::shortestParlindrome(std::string s)
{
	std::string res;
	for (int i = s.length() - 1; i >= 0; i--)
	{
		if (isPalindrome(s.substr(0, i)))
		{
			res = std::string(s.begin() + i, s.end());
			res.reserve();
			return res + s;
		}
	}
	res = s;
	res.reserve();
	return res + s;
}

std::vector<int> Solution::pancakeSort(std::vector<int>& A)
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
	std::vector<int>  subA = std::vector<int>(A.rbegin(), A.rbegin() + A.size() - idx);
	for (size_t i = 0; i + 1 < idx; i++)
		subA.push_back(A[i]);
	auto follows = pancakeSort(subA);
	for (const auto& num : follows)
		res.push_back(num);
	return res;
}

bool Solution::containsPattern(std::vector<int>& arr, int m, int k)
{

	return false;
}

int Solution::getMaxLen(std::vector<int>& nums)
{
	auto helper = [](std::vector<int>& nums) ->int {
		auto temp = 1;
		for (auto& num : nums)
		{
			temp *= std::abs(num) / num;
		}
		if (temp == 1) return nums.size();
		int i = 0;
		while (0 < nums[i] && 0 < nums[nums.size() - 1 - i])
			i++;
		return  -i - 1 + nums.size();
	};
	nums = { 0,1,-2,-3,-4 };
	std::vector<std::vector<int>> withoutzero;
	std::unordered_map <int, std::vector<int>> umap;
	int start = 0;
	for (size_t i = 0; i < nums.size(); i++)
	{
		if (nums[i] == 0) {
			withoutzero.push_back(std::vector<int>(nums.begin() + start, nums.begin() + i));
			start = i + 1;
		}
	}
	withoutzero.push_back(std::vector<int>(nums.begin() + start, nums.end()));
	int res = 0;
	for (auto& interval : withoutzero)
		res = std::max(helper(interval), res);
	return res;
}

// TODO: TLE
int Solution::largestComponentSize(std::vector<int>& A)
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

TreeNode* removeAt(TreeNode*& root) {
	TreeNode* curr = root;
	TreeNode* succ = nullptr;
	if (root->left == nullptr)
		root = root->right;
	else if (root->right == nullptr)
		root = root->left;
	else
	{
		TreeNode* target = curr;
		TreeNode* parent = nullptr;
		curr = curr->right;
		while (curr->left != nullptr)
		{
			parent = curr;
			curr = curr->left;
		}
		std::swap(curr->val, target->val);
		(parent == root ? parent->left : parent->right) = curr->right;
	}
	return succ;
}

TreeNode* Solution::deleteNode(TreeNode* root, int key)
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

	TreeNode* curr = root;
	TreeNode* parent = nullptr;

	while (curr != nullptr)
	{
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

	if (curr->left == nullptr)
	{
		if (curr->right != nullptr)
			*curr = *curr->right;
		else
			curr == parent->left ? parent->left = nullptr : parent->right = nullptr;
	}
	else if (curr->right == nullptr)
	{
		*curr = *curr->left;
	}
	else {
		TreeNode* succ = curr->right;
		parent = curr;
		while (succ->left != nullptr)
		{
			parent = succ;
			succ = succ->left;
		}
		std::swap(curr->val, succ->val);
		if (succ->right != nullptr)
			*succ = *succ->right;
		else
			succ == parent->left ? parent->left = nullptr : parent->right = nullptr;
	}

	return root;
}

// TODO:
std::string Solution::largestTimeFromDigits(std::vector<int>& A)
{
	std::string result;
	std::vector<int> B(4, 0);
	std::sort(A.begin(), A.end());
	if (2 < A[0])
		return result;
	if (5 < A[1])
		return result;
	if (2 == A[0])
	{
		if (3 < A[1])
			return result;
		if (5 < A[2])
			return result;
		if (A[2] < 4)
		{
			B[0] = 2;
			//B[1] == 
		}
	}
	else
	{

	}

	return result;
}

std::vector<std::vector<int>> Solution::permute(std::vector<int>& nums)
{
	if (nums.size() < 2)
		return std::vector<std::vector<int>>(1, nums);

	std::vector<std::vector<int>> result;
	for (size_t i = 0; i < nums.size(); i++)
	{
		std::vector<int> except_i = nums;
		except_i.erase(except_i.begin() + i);
		for (auto permutation : permute(except_i))
		{
			permutation.push_back(nums[i]);
			result.emplace_back(permutation);
		}
	}
	return result;
}

bool Solution::containsNearbyAlmostDuplicate(std::vector<int>& nums, int k, int t)
{
	std::map<int, std::vector<int>> map;
	for (size_t i = 0; i < nums.size(); i++)
		map[nums[i]].push_back(i);
	for (auto& e : map)
		for (size_t i = 0; i < e.second.size(); i++)
			for (size_t minus = 0; minus <= t; minus++)
				for (size_t j = 0; j < map[e.first + minus].size(); j++)
					if (e.second[i] != map[e.first + minus][j] && std::abs(e.second[i] - map[e.first + minus][j]) <= k)
						return true;
	return false;
}

std::vector<std::string> Solution::binaryTreePaths(TreeNode* root)
{
	const std::string arrow = "->";

	static auto dfs = make_y_combinator([&](auto&& dfs, TreeNode* root, std::string str, std::vector<std::string>& result)->void {
		if (root == nullptr)
			return;
		str.append(std::to_string(root->val));
		if (root->left == nullptr && root->right == nullptr) {
			result.push_back(str);
		}
		else {
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
	while (left < S.length())
	{
		right = S.length();
		while (S[--right] != S[left])
			;
		flag[S[left] - 'a'] = false;
		while (left < right)
		{
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

std::vector<int> Solution::getAllElements(TreeNode* root1, TreeNode* root2)
{
	auto first = inorderTraversal(root1);
	auto second = inorderTraversal(root2);
	std::vector<int> res(first.size() + second.size(), 0);

	// merge
	size_t i = 0, j = 0, k = 0;
	while (j < first.size() && k < second.size())
	{
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
	for (size_t i = 0; i < s.length(); i++)
	{
		if (s[i] == '?')
		{
			char temp = 'a';
			if (i == 0)
			{
				while (temp == s[i + 1])
					temp = letterAfter(temp);
			}
			else if (i == s.length() - 1)
			{
				while (temp == s[i - 1])
					temp = letterAfter(temp);
			}
			else
			{
				while (temp == s[i + 1] || temp == s[i - 1])
					temp = letterAfter(temp);
			}
			s[i] = temp;
		}
	}
	return s;
}

int Solution::minCost(std::string s, std::vector<int>& cost)
{
	int res = 0;
	for (size_t i = 1; i < s.length(); i++)
	{
		if (s[i] == s[i - 1])
		{
			if (cost[i] < cost[i - 1])
				std::swap(cost[i], cost[i - 1]);
			res += cost[i - 1];
		}
	}
	return res;
}

int Solution::numTriplets(std::vector<int>& nums1, std::vector<int>& nums2)
{
	//nums1 = { 3, 1, 2, 2 };
	//nums2 = { 1, 3, 4, 4 };
	//nums1 = { 7, 3, 4, 2, 1, 4, 1, 6, 1, 1, 5 };
	//nums2 = { 3, 5, 2, 4, 3, 1, 7, 5, 7, 5 }; 
	//nums1 = { 3, 5, 1, 2, 4, 3, 3, 2, 4, 2, 3, 4, 5, 2, 4, 3, 5, 3, 4, 5, 3, 1, 1, 2, 4, 2, 4, 1, 2, 1, 2, 2, 5, 2, 4, 5, 4, 5, 5, 2, 4, 4, 5, 3, 1, 2, 5, 4, 5, 1, 2 };
	//nums2 = { 3, 4, 4, 3, 3, 5, 5, 4, 3, 3, 1, 4, 5, 4, 2, 4, 2, 2, 2, 5, 4, 4, 4, 5, 2, 4, 2, 1, 2, 5, 2, 5, 5, 3, 5, 4, 3, 4, 3, 5, 1, 1, 4, 5, 3, 1, 5, 5, 5, 2, 5, 4, 2, 4, 5, 3, 2, 2 };
	auto choose2 = [](int n) {
		return n * (n - 1) / 2;
	};
	int res = 0;
	std::map<int, int> map1, map2;
	for (size_t i = 0; i < nums1.size(); i++)
		map1[nums1[i]]++;
	for (size_t i = 0; i < nums2.size(); i++)
		map2[nums2[i]]++;
	for (auto e : map1)
	{
		if (1 < map2[e.first])
			res += e.second * choose2(map2[e.first]);
		if (1 < map2.size())
		{
			long squre = static_cast<long>(e.first) * e.first;
			auto left = map2.begin();
			auto right = map2.rbegin();
			while ((*left).first < e.first && e.first < (*right).first)
			{
				long product = static_cast<long>((*left).first) * (*right).first;
				if (product < squre)
					left++;
				else if (squre < product)
					right++;
				else {
					res += e.second * (*left).second * (*right).second;
					left++; right++;
				}
			}
		}
	}
	for (auto e : map2)
	{
		if (1 < map1[e.first])
			res += e.second * choose2(map1[e.first]);
		if (1 < map1.size())
		{
			long squre = static_cast<long>(e.first) * e.first;
			auto left = map1.begin();
			auto right = map1.rbegin();
			while ((*left).first < e.first && e.first < (*right).first)
			{
				long product = static_cast<long>((*left).first) * (*right).first;
				if (product < squre)
					left++;
				else if (squre < product)
					right++;
				else {
					res += e.second * (*left).second * (*right).second;
					left++; right++;
				}
			}
		}
	}
	return res;
}

int Solution::maxNumEdgesToRemove(int n, std::vector<std::vector<int>>& edges)
{
	return 0;
}

int Solution::largestOverlap(std::vector<std::vector<int>>& A, std::vector<std::vector<int>>& B)
{
	return 0;
}

std::vector<int> Solution::dailyTemperatures(std::vector<int>& T)
{
	int length = T.size();
	std::vector<int> res(length, 0);
	std::vector<int> hash(71, length);
	for (int i = T.size() - 1; i >= 0; i--)
	{
		int min_value = length;
		for (int temp = T[i] + 1; temp <= 100; temp++)
		{
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

int Solution::findTargetSumWays(std::vector<int>& nums, int S)
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
		dp[std::make_pair(subnums, S - nums.back())] = findTargetSumWays(subnums, S - nums.back());
	if (dp.find(std::make_pair(subnums, S + nums.back())) == dp.end())
		dp[std::make_pair(subnums, S + nums.back())] = findTargetSumWays(subnums, S + nums.back());
	return dp[std::make_pair(subnums, S - nums.back())] + dp[std::make_pair(subnums, S + nums.back())];
}

int Solution::subarraySum(std::vector<int>& nums, int k) {
	int res = 0;
	int sum = 0;
	std::unordered_map<int, int> umap;
	for (size_t i = 0; i < nums.size(); i++)
	{
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
	for (int i = 1; i < s.length() - 1; i++)
	{
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
			while (0 <= i - k - 1 && i + k < s.length() && s[i - k - 1] == s[i + k])
				k++;
			res += k - 1;
		}
	return res;
}

bool Solution::wordPattern(std::string pattern, std::string str)
{
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
	std::vector<std::string> word_list = split(str, ' ');
	if (word_list.size() == pattern.length())
	{
		std::unordered_map<char, std::string> hash;
		std::unordered_map<std::string, char> rhash;
		for (size_t i = 0; i < word_list.size(); i++) {
			if (hash.find(pattern[i]) == hash.end() && rhash.find(word_list[i]) == rhash.end())
			{
				hash[pattern[i]] = word_list[i];
				rhash[word_list[i]] = pattern[i];
			}
			else if (hash[pattern[i]] != word_list[i] || rhash[word_list[i]] != pattern[i])
				return false;
		}
		return true;
	}
	return false;
}

std::vector<std::vector<int>> Solution::combine(int n, int k)
{
	auto bits = [](long num) -> int {
		int count = 0;
		while (num != 0) {
			num &= num - 1;
			count++;
		}
		return count;
	};
	std::vector<std::vector<int>> res;
	for (long i = 0; i < 1 << n; i++)
	{
		std::vector<int> temp;
		if (bits(i) == k)
		{
			for (size_t bit_index = 0; bit_index < n; bit_index++)
			{
				if (1 == 1 & (i >> bit_index))
					temp.push_back(bit_index + 1);
			}
		}
		res.push_back(temp);
	}
	return res;
}

int Solution::sumRootToLeaf(TreeNode* root)
{
	int sum = 0;
	static auto dfs = make_y_combinator([&](auto&& dfs, TreeNode* node, int num) -> void {
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

auto lambda = [](const std::vector<int>& candidates, const std::vector<int>& nums) {
	int res = 0;
	for (size_t i = 0; i < nums.size(); i++)
		res += nums[i] * candidates[i];
	return res;
};
// TODO:
void backtrack(std::vector<int>& candidates, int target, int current_index, std::vector<int> num_of_times, std::vector<std::vector<int>>& result)
{
	if (candidates.size() <= current_index) //oversize()
		return;
	for (size_t i = 0; ; i++) {
		num_of_times[current_index] = i;

		int sum = lambda(candidates, num_of_times);
		if (sum < target) // noconfilct
			backtrack(candidates, target, current_index + 1, num_of_times, result);
		else if (sum == target)
		{
			std::vector<int> combination;
			for (size_t j = 0; j < num_of_times.size(); j++)
			{
				int time = 0;
				while (num_of_times[j] - time++ != 0)
					combination.push_back(candidates[j]);
			}
			if (!combination.empty())
				result.push_back(combination);
		}
		else
			return;
		num_of_times[current_index] = 0;
	}
}
std::vector<std::vector<int>> Solution::combinationSum(std::vector<int>& candidates, int target)
{
	std::vector<std::vector<int>> result;
	std::vector<int> times(candidates.size(), 0);
	backtrack(candidates, target, 0, times, result);
	return result;
}

int Solution::numBusesToDestination(std::vector<std::vector<int>>& routes, int S, int T)
{
	return 0;
}
std::vector<int> distanceKdescendants(TreeNode* root, TreeNode* except, int K);
std::vector<int> Solution::distanceK(TreeNode* root, TreeNode* target, int K)
{
	std::unordered_map<TreeNode*, TreeNode*> umap;
	auto dfs = make_y_combinator([&](auto&& dfs, TreeNode* root, TreeNode* target) ->void {
		if (root == nullptr || root == target) return;
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
	for (auto& e : umap)
		std::cout << "node " << e.first->val << "'s parent is node" << e.second->val << std::endl;
	std::vector<int> res;

	std::queue<TreeNode*> que;
	que.push(target);
	while (!que.empty() && 0 <= K)
	{
		int count = que.size();
		std::cout << K << std::endl;
		while (0 != count--)
		{
			TreeNode* curr = que.front();
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

int Solution::maxPathSum(TreeNode* root)
{

	return 0;
}

int Solution::maxProduct(std::vector<int>& nums)
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

std::vector<std::vector<int>> Solution::combinationSum3(int k, int n)
{
	std::vector<std::vector<int>> result;
	return result;
}

std::vector<double> Solution::averageOfLevels(TreeNode* root)
{
	std::vector<double> result;
	if (root == nullptr) return result;
	std::queue<TreeNode*> queue;
	queue.push(root);
	int count = 0;
	std::vector<std::vector<int>> level_order;
	while (!queue.empty())
	{
		count = queue.size();
		std::vector<int> level;
		int sum_of_level = 0;
		int size = 0;
		while (0 != count--)
		{
			TreeNode* curr = queue.front();
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

int Solution::numSpecial(std::vector<std::vector<int>>& mat)
{
	int result = 0;
	int n = mat.size();
	std::vector<int> sum_of_row(n, 0);
	std::vector<int> sum_of_colunm(n, 0);
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			sum_of_row[i] += mat[i][j];
			sum_of_colunm[i] += mat[j][i];
		}
	}
	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < n; j++)
			if (mat[i][j] == 1 && sum_of_row[i] == 1 && sum_of_colunm[j] == 1)
				result++;
	return result;
}

int Solution::unhappyFriends(int n, std::vector<std::vector<int>>& preferences, std::vector<std::vector<int>>& pairs)
{
	int res = 0;
	std::vector<bool> unhappy(n, false);
	std::map<int, int> pair_map;
	for (auto pair : pairs) {
		pair_map[pair[0]] = pair[1];
		pair_map[pair[1]] = pair[0];
	}
	for (int i = 0; i < n; i++)
	{
		if (unhappy[i] == false) {
			for (size_t j = 0; j < preferences[i].size(); j++)
			{
				if (preferences[i][j] == pair_map[i])
					break;
				//check preferences[pair[0]][i] happy
				for (size_t k = 0; k < preferences[j].size(); k++)
				{
					if (preferences[preferences[i][j]][k] == pair_map[preferences[i][j]])
						break;
					if (preferences[preferences[i][j]][k] == i) {
						unhappy[i] = true;
						unhappy[preferences[i][j]] = true;
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
int Solution::minCostConnectPoints(std::vector<std::vector<int>>& points)
{
	static auto manhattan_distance = [](std::vector<int>& pointA, std::vector<int>& pointB) {
		return std::abs(pointA[0] - pointB[0]) + std::abs(pointA[1] - pointB[1]);
	};
#if false
	// Brute Greedy : TLE
	int res = 0;
	int connected_count = 1;
	std::vector<bool> connected(points.size(), false);
	connected[0] = true;
	while (connected_count < points.size()) {
		int min_distance = INT_MAX;
		int idx = 0;
		for (size_t i = 0; i < points.size(); i++) {
			if (!connected[i]) continue;
			for (size_t j = 0; j < points.size(); j++) {
				if (connected[j] || j == i) continue;
				if (manhattan_distance(points[i], points[j]) < min_distance) {
					min_distance = manhattan_distance(points[i], points[j]);
					idx = j;
				}
			}
		}
		res += min_distance;
		connected[idx] = true;
		connected_count++;
	}
#elif true
	// Kruskal's
	struct edge
	{
		int v;
		int w;
		int weight;
		edge(int _v, int _w, int _weight) :v(_v), w(_w), weight(_weight) {}
		bool operator< (edge other) {
			return weight < other.weight;
		}
		bool operator> (edge other) {
			return weight > other.weight;
		}
	};
	int res = 0;
	std::vector<edge> edges;
	for (size_t i = 0; i + 1 < points.size(); i++)
	{
		for (size_t j = i + 1; j < points.size(); j++)
		{
			edges.emplace_back(i, j ,manhattan_distance(points[i], points[j]));
		}
	}
	std::sort(edges.begin(), edges.end());
	for (const auto& e : edges)
	{
		// union-find (e.v,e.w)
		// if not connected
		{
			res += e.weight;
			//update union find;
		}
	}
#else
	// Prim's

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
	if (sorted_s != sorted_t) return false;

	int count[2] = { 0 };
	for (size_t i = 0; i + 1 < s.length(); i++)
	{
		for (size_t j = i + 1; j < s.length(); j++)
		{
			if (s[j] < s[i])
				count[0]++;
			if (t[j] < t[i])
				count[1]++;
		}
	}
	return count[1] < count[0];
}

std::vector<std::vector<int>> Solution::insert(std::vector<std::vector<int>>& intervals, std::vector<int>& newInterval)
{
	std::vector<std::vector<int>> result;
	for (auto interval : intervals)
	{
		if (interval[1] < newInterval[0] || newInterval[1] < interval[0])
			result.push_back(interval);
		else {
			newInterval[0] = std::min(newInterval[0], interval[0]);
			newInterval[1] = std::max(newInterval[1], interval[1]);
		}
	}
	auto it = result.begin();
	for (; ; it++)
	{
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
		}
		else if ((a >> i) & 1 || (b >> i) & 1) {
			if (carry == 0)
				res |= 1 << i;
		}
		else {
			res |= carry << i;
			carry = 0;
		}
	}
	return res;
}

int Solution::strStr(std::string haystack, std::string needle)
{
	// brute force
	for (size_t i = 0; i < haystack.size() - needle.size() + 1; i++)
	{
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
int Solution::findMaximumXOR(std::vector<int>& nums)
{
	struct TrieNode
	{
		std::map<int, TrieNode*> children;
		bool is_end = false;
	};
	// findbits
	int max_num = 0;
	for (auto num : nums)
	{
		if (max_num < num)
			max_num = num;
	}
	int bit_count = 0;
	while (max_num >> bit_count != 0)
		bit_count++;

	//insert to Trie
	TrieNode* root = new TrieNode();
	TrieNode* curr = root;
	for (auto num : nums)
	{
		//insert
		curr = root;
		for (int i = bit_count - 1; i >= 0; i--)
		{
			auto bit = (num >> i) & 1;
			if (curr->children.find(bit) == curr->children.end()) {
				TrieNode* node = new TrieNode();
				curr->children[bit] = node;
			}
			curr = curr->children[bit];
		}
		curr->is_end = true;
	}

	int res = 0;
	for (auto num : nums)
	{
		if ((1 << (bit_count - 1)) <= num) {
			curr = root;
			int toXor = 0;
			for (int i = bit_count - 1; i >= 0; i--) {
				auto bit = (num >> i) & 1;
				toXor <<= 1;
				if (curr->children.find(1 - bit) != curr->children.end()) {
					curr = curr->children[1 - bit];
					toXor += 1 - bit;
				}
				else {
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
	struct coord
	{
		int x = 0;
		int y = 0;
		void operator +=(coord other) {
			x += other.x;
			y += other.y;
		}
		bool operator ==(coord other) {
			return x == other.x && y == other.y;
		}
		void rotate(double rad) {
			auto temp = std::cos(rad) * x - std::sin(rad) * y;
			y = std::sin(rad) * x + std::cos(rad) * y;
			x = temp;
		}
	};
	coord position = { 0, 0 };
	coord direction = { 0, 1 };
	for (auto letter : instructions)
	{
		if (letter == 'G')
		{
			//move_along_direction;
			position += direction;
		}
		else
		{
			//change direction
			if (letter == 'L')
				direction.rotate(-pi / 2.0);
			else
				direction.rotate(pi / 2.0);
		}
	}

	return direction.x != 0 || direction.y != 1 || (position.x == 0 && position.y == 0);
}

int Solution::countPrimes(int n)
{
	int count = 0;
	std::vector<bool> is_prime(n + 1, true);
	for (size_t i = 2; i < n + 1; i++) {
		if (is_prime[i] == false)
			continue;
		auto num = i + i;
		while (num < n + 1)
		{
			if (is_prime[num])
				is_prime[num] = false;
			num += i;
		}
	}
	for (size_t i = 1; i < n + 1; i++)
	{
		if (is_prime[i])
			count++;
	}
	return count;
}

std::string Solution::simplifyPath(std::string path)
{
	static auto split = [](const std::string& str, const char& chr) -> std::vector<std::string> {
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
	std::vector<std::string> directorys = split(path, '/');
	std::vector<std::string> s;
	std::string result;
	for (auto r : directorys) {
		if (r == "..") {
			if (!s.empty())
				s.resize(s.size() - 1);
		}
		else if (r == ".")
			;
		else {
			s.push_back(r);
		}
	}
	for (auto r : s)
		result.append("/" + r);
	return result;
}

void dfs(TreeNode* root, std::vector<std::vector<std::string>>& grid, int row, int column) {
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
std::vector<std::vector<std::string>> Solution::printTree(TreeNode* root)
{
	static auto height = make_y_combinator([](auto&& height, TreeNode* node) -> int {
		if (node == nullptr) return -1;
		return std::max(height(node->left), height(node->right)) + 1;
		});
	static auto print_helper = make_y_combinator([](auto&& print_helper, TreeNode* root, std::vector<std::vector<std::string>>& grid, int row, int column, int delta_column) -> void {
		if (nullptr == root)
			return;
		grid[row][column] = std::to_string(root->val);
		row++;
		delta_column >>= 1;
		print_helper(root->left, grid, row, column - delta_column, delta_column);
		print_helper(root->right, grid, row, column + delta_column, delta_column);
		}
	);
	int height_of_root = height(root);
	std::vector<std::vector<std::string>> result(height_of_root + 1, std::vector<std::string>((1 << (height_of_root + 1)) - 1, ""));
	print_helper(root, result, 0, (1 << height_of_root) - 1, 1 << height_of_root);
	return result;
}

bool Solution::isCompleteTree(TreeNode* root)
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
	std::string res;
	std::vector<std::string> words = split(text, ' ');
	int len = text.length();
	for (auto& word : words)
		len -= word.length();
	int per = len;
	if (1 < words.size())
		per /= words.size() - 1;
	for (auto& word : words) {
		res.append(word);
		for (size_t i = 0; i < per; i++)
			res.push_back(' ');
	}
	res.resize(len);
	return res;
}
int Solution::maxUniqueSplit(std::string s)
{
	std::map<std::string, bool> apmap;
	std::vector<std::string> strs;
	int count = 0;
	for (size_t i = 0; i < s.length(); )
	{
		for (size_t j = 1; i + j <= s.length(); j++)
		{
			auto str = s.substr(i, j);
			if (apmap.end() == apmap.find(str) || false == apmap[str])
			{
				apmap[str] = true;
				strs.emplace_back(str);
				i += j;
				count++;
				break;
			}
			else {
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
void dfs(std::vector<std::vector<int>>& grid, long value, int row, int column, int row_size, int column_size) {
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
int Solution::maxProductPath(std::vector<std::vector<int>>& grid)
{
	max_value = -1;
	dfs(grid, 1, 0, 0, grid.size(), grid[0].size());
	if (1000000007 < max_value)
		max_value %= 1000000007;
	return max_value;
}
