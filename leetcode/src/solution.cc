#include "solution.h"

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

bool Solution::isSubsequence(std::string s, std::string t)
{
    s = "abc", t = "ahbgdc";
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
        //auto k = std::find(t.begin(),t.end(),*it);
        ////t.reserve();
        //if (t.find(*it) == -1)
        //    return false;
        //t.erase(t.reserve().find(*it));
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
}

// TODO:
int Solution::hIndex(std::vector<int>& citations)
{
    int N = citations.size();
    int left = 0, right = N;
    int res = (left + right) / 2;
    while (left < right)
    {
        if (citations[res - 1] < res)
            left = res;
        if (res <= citations[res - 1])
            right = res;
        res = (left + right) / 2;
    }
    return res;
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
#if recursive
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
    {
        for (size_t j = i + 1; j < arr.size() - 1; j++)
        {
            for (size_t k = j + 1; k < arr.size(); k++)
            {
                if ((arr[i] - arr[j] <= a) && (-a <= arr[i] - arr[j]) && (arr[j] - arr[k] <= b) && (-b <= arr[j] - arr[k]) && (arr[i] - arr[k] <= c) && (-c <= arr[i] - arr[k]))
                {
                    count++;
                }
            }
        }
    }
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
