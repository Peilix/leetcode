#include "solution.h"
#include <queue>

#include "y_combinator.h"

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
    auto dfs = make_y_combinator([](auto&& dfs, int row,int column, std::vector<std::vector<char>> grid) {
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
    std::sort(citations.begin(), citations.end(),std::greater<int>());

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
    while(n)
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
        if (grid[0].back()==0)
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
    return 0;
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
            q.push(std::make_pair(node->left,x-1));
        if (node->right)
            q.push(std::make_pair(node->right,x+1));
    }
    for (auto e:map)
    {
        res.push_back(e.second);
    }
    return res;
}

std::vector<std::vector<int>> Solution::verticalTraversal(TreeNode* root)
{
    std::vector<std::vector<int>> res;
    std::queue<std::pair<TreeNode*, std::pair<int,int>>> q;
    std::map<int, std::vector<std::pair<int,int>>> map;
    q.push(std::make_pair(root, std::make_pair(0, 0)));
    while (!q.empty())
    {
        auto node = q.front().first;
        auto x = q.front().second.first;
        auto y = q.front().second.second;
        auto val = node->val;
        if (0 == map.count(x))
            map[x] = std::vector<std::pair<int, int>>();
        map[x].push_back(std::make_pair(node->val,y));
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
                res.push_back(curr->val);
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
    auto helper = make_y_combinator([](auto&& height, TreeNode* root)->int {
        if (root == nullptr)
            return 0;
        int leftHeight = 0, rightHeight = 0;
        if (-1 == (leftHeight = helper(root->left)) || -1 == (rightHeight = helper(root->right)) || 1 < abs(leftHeight - rightHeight))
            return -1;
        else
            return std::max(leftHeight, rightHeight) + 1;
        });
    return 0 <= helper(root);
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
