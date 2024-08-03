# LeetCode 热题 100

## 哈希

[1. 两数之和](https://leetcode.cn/problems/two-sum/)
```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> table;
        for (int i = 0; i < nums.size(); i++) {
            int l = target - nums[i];
            if (table.find(l) != table.end()) {
                return {table[l], i};
            }
            table[nums[i]] = i;
        }
        return {};
    }
};
```

[49. 字母异位词分组](https://leetcode-cn.com/problems/group-anagrams/)
> 给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。  
> 字母异位词 是由重新排列源单词的所有字母得到的一个新单词。  
> 输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]   
> 输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
```c++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> table;
        for (const auto &str : strs) {
            string key = str;
            sort(key.begin(), key.end());
            table[key].emplace_back(str);
        }
        vector<vector<string>> ans;
        for (const auto &item : table) {
            ans.emplace_back(item.second);
        }
        return ans;
    }
};
```

[128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)
> 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。  
> 请你设计并实现时间复杂度为 O(n) 的算法解决此问题。
```c++
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> table(nums.begin(), nums.end());
        int ans = 0;
        for (const auto &num : nums) {
            if (table.find(num - 1) == table.end()) {
                int end = num;
                while (table.find(end + 1) != table.end()) {
                    end++;
                }
                ans = max(ans, end - num + 1);
            }
        }
        return ans;
    }
};
```

## 双指针

[283. 移动零](https://leetcode-cn.com/problems/move-zeroes/)
> 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。  
> 请注意 ，必须在不复制数组的情况下原地对数组进行操作。
```c++
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int k = 0;
        for (const auto &num : nums) {
            if (num != 0) {
                nums[k++] = num;
            }
        }
        for (int i = k; i < nums.size(); i++) {
            nums[i] = 0;
        }
    }
};
```

[11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)
> 给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。  
> 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
```c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int left = 0, right = height.size() - 1;
        int leftMax = 0, rightMax = 0;
        int ans = 0;
        while (left < right) {
            leftMax = max(leftMax, height[left]);
            rightMax = max(rightMax, height[right]);
            if (leftMax < rightMax) {
                ans = max(ans, leftMax * (right - left));
                left++;
            } else {
                ans = max(ans, rightMax * (right - left));
                right--;
            }
        }
        return ans;
    }
};
```

[15. 三数之和](https://leetcode-cn.com/problems/3sum/)
```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> ans;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size(); i++) {
            if (i && nums[i] == nums[i - 1]) {
                continue;
            }
            for (int j = i + 1, k = nums.size() - 1; j < k; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) {
                    continue;
                }
                while (j < k - 1 && nums[i] + nums[j] + nums[k - 1] >= 0) {
                    k--;
                }
                if (nums[i] + nums[j] + nums[k] == 0) {
                    ans.push_back({nums[i], nums[j], nums[k]});
                }
            }
        }
        return ans;
    }
};
```

[209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)
> 给定一个含有 n 个正整数的数组和一个正整数 target。
> 找出该数组中满足其总和大于等于 target 的长度最小的子数组，并返回其长度。如果不存在符合条件的子数组，返回 0。
```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int ans = INT_MAX;
        int sum = 0;
        for (int i = 0, j = 0; i < nums.size(); i++) {
            sum += nums[i];
            while (sum - nums[j] >= target) {
                sum -= nums[j++];
            }
            if (sum >= target) {
                ans = min(ans, i - j + 1);
            }
        }
        return ans == INT_MAX ? 0 : ans;
    }
};
```

[42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)
> 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
```c++
class Solution {
public:
    int trap(vector<int>& height) {
        int left = 0, right = height.size() - 1;
        int leftMax = 0, rightMax = 0;
        int ans = 0;
        while (left < right) {
            leftMax = max(leftMax, height[left]);
            rightMax = max(rightMax, height[right]);
            if (leftMax < rightMax) {
                ans += leftMax - height[left];
                left++;
            } else {
                ans += rightMax - height[right];
                right--;
            }
        }
        return ans;
    }
};
```

## 滑动窗口

[3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)
```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char, int> sCount;
        int ans = 0;
        for (int i = 0, j = 0; i < s.size(); i++) {
            sCount[s[i]]++;
            while (sCount[s[i]] > 1) {
                sCount[s[j++]]--;
            }
            ans = max(ans, i - j + 1);
        }
        return ans;
    }
};
```

[438. 找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)
> 给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。  
> 异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。  
> 输入: s = "cbaebabacd", p = "abc"  
> 输出: [0,6]
```C++
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        vector<int> ans;
        if (s.size() < p.size()) {
            return ans;
        }
        vector<int> sCount(26);
        vector<int> pCount(26);
        for (int i = 0; i < p.size(); i++) {
            sCount[s[i] - 'a']++;
            pCount[p[i] - 'a']++;
        }
        if (sCount == pCount) {
            ans.emplace_back(0);
        }
        for (int i = 1; i < s.size() - p.size() + 1; i++) {
            sCount[s[i - 1] - 'a']--;
            sCount[s[i + p.size() - 1] - 'a']++;
            if (sCount == pCount) {
                ans.emplace_back(i);
            }
        }
        return ans;
    }
};
```

## 子串

[560. 和为 K 的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)
> 输入：nums = [1,1,1], k = 2  
> 输出：2
```c++
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        vector<int> sum(nums.size());
        sum[0] = nums[0];
        for (int i = 1; i < sum.size(); i++) {
            sum[i] = sum[i - 1] + nums[i];
        }
        unordered_map<int, int> table;
        table[0] = 1;
        int ans = 0;
        for (int i = 0; i < nums.size(); i++) {
            ans += table[sum[i] - k];
            table[sum[i]]++;
        }
        return ans;
    }
};
```

[239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)
> 输入：nums = [1,3,-1,-3,5,3,6,7], k = 3  
> 输出：[3,3,5,5,6,7]
```c++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ans;
        deque<int> q;
        for (int i = 0; i < nums.size(); i++) {
            if (q.size() && q.front() < i - k + 1) {
                q.pop_front();
            }
            while (q.size() && nums[i] >= nums[q.back()]) {
                q.pop_back();
            }
            q.push_back(i);
            if (i >= k - 1) {
                ans.emplace_back(nums[q.front()]);
            }
        }
        return ans;
    }
};
```

[76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)
> 输入：s = "ADOBECODEBANC", t = "ABC"
> 输出："BANC"
```c++
class Solution {
public:
    string minWindow(string s, string t) {
        unordered_map<char, int> sCount, tCount;
        for (const auto &ch : t) {
            tCount[ch]++;
        }
        int cnt = 0;
        string ans;
        for (int i = 0, j = 0; i < s.size(); i++) {
            sCount[s[i]]++;
            if (sCount[s[i]] <= tCount[s[i]]) {
                cnt++;
            }
            while (sCount[s[j]] > tCount[s[j]]) {
                sCount[s[j++]]--;
            }
            if (cnt == t.size()) {
                if (ans.empty() || i - j + 1 < ans.size()) {
                    ans = s.substr(j, i - j + 1);
                }
            }
        }
        return ans;
    }
};
```

## 普通数组

[53. 最大子数组和](https://leetcode-cn.com/problems/maximum-subarray/)
> 输入：nums = [-2,1,-3,4,-1,2,1,-5,4]  
> 输出：6  
> 解释：连续子数组 [4,-1,2,1] 的和最大，为 6。
```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int ans = INT_MIN;
        for (int i = 0, last = 0; i < nums.size(); i++) {
            last = nums[i] + max(last, 0);
            ans = max(ans, last);
        }
        return ans;
    }
};
```

[56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)
> 输入：intervals = [[1,3],[2,6],[8,10],[15,18]]  
> 输出：[[1,6],[8,10],[15,18]]
```c++
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>> ans;
        if (intervals.empty()) {
            return ans;
        }
        sort(intervals.begin(), intervals.end());
        int left = intervals[0][0], right = intervals[0][1];
        for (int i = 1; i < intervals.size(); i++) {
            if (intervals[i][0] <= right) {
                right = max(right, intervals[i][1]);
            } else {
                ans.push_back({left, right});
                left = intervals[i][0];
                right = intervals[i][1];
            }
        }
        ans.push_back({left, right});
        return ans;
    }
};
```

[189. 轮转数组](https://leetcode-cn.com/problems/rotate-array/)
> 输入: nums = [1,2,3,4,5,6,7], k = 3  
> 输出: [5,6,7,1,2,3,4]
```c++
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        k = k % nums.size();
        reverse(nums.begin(), nums.end());
        reverse(nums.begin(), nums.begin() + k);
        reverse(nums.begin() + k, nums.end());
    }
};
```

[238. 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)
```c++
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> forward(nums.size(), 1);
        for (int i = 1; i < nums.size(); i++) {
            forward[i] = nums[i - 1] * forward[i - 1];
        }
        for (int i = nums.size() - 1, backward = 1; i >= 0; i--) {
            forward[i] = forward[i] * backward;
            backward = backward * nums[i];
        }
        return forward;
    }
};
```

[41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)
> 给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。  
> 请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。  
> 输入：nums = [3,4,-1,1]  
> 输出：2
```c++
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            while (nums[i] >= 1 && nums[i] <= n && nums[i] != nums[nums[i] - 1]) {
                swap(nums[i], nums[nums[i] - 1]);
            }
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }
};
```

## 矩阵

[73. 矩阵置零](https://leetcode-cn.com/problems/set-matrix-zeroes/)
> 给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
```c++
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        bool row0 = false, col0 = false;

        for (int i = 0; i < m; i++) {
            if (!matrix[i][0]) {
                col0 = true;
            }
        }

        for (int j = 0; j < n; j++) {
            if (!matrix[0][j]) {
                row0 = true;
            }
        }

        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!matrix[i][j]) {
                    matrix[i][0] = 0;
                }
            }
        }

        for (int j = 1; j < n; j++) {
            for (int i = 0; i < m; i++) {
                if (!matrix[i][j]) {
                    matrix[0][j] = 0;
                }
            }
        }

        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!matrix[i][0]) {
                    matrix[i][j] = 0;
                }
            }
        }

        for (int j = 1; j < n; j++) {
            for (int i = 0; i < m; i++) {
                if (!matrix[0][j]) {
                    matrix[i][j] = 0;
                }
            }
        }

        for (int i = 0; i < m; i++) {
            if (col0) {
                matrix[i][0] = 0;
            }
        }

        for (int j = 0; j < n; j++) {
            if (row0) {
                matrix[0][j] = 0;
            }
        }
    }
};
```

[54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)
> 给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
```c++
class Solution {
    vector<int> dx = {0, 1, 0, -1};
    vector<int> dy = {1, 0, -1, 0};
    vector<int> ans;
    vector<vector<bool>> visited;

public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        visited = vector<vector<bool>>(m, vector<bool>(n));
        for (int x = 0, y = 0, dir = 0, i = 0; i < m * n; i++) {
            ans.emplace_back(matrix[x][y]);
            visited[x][y] = true;
            int a = x + dx[dir];
            int b = y + dy[dir];
            if (a < 0 || a >= m || b < 0 || b >= n || visited[a][b]) {
                dir = (dir + 1) % dx.size();
                a = x + dx[dir];
                b = y + dy[dir];
            }
            x = a, y = b;
        }
        return ans;
    }
};
```

[48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)
> 给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。  
> 你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
```c++
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < i; j++) {
                swap(matrix[i][j], matrix[j][i]);
            }
        }

        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix.size() / 2; j++) {
                swap(matrix[i][j], matrix[i][matrix.size() - j - 1]);
            }
        }
    }
};
```

[240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)
> 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
> - 每行的元素从左到右升序排列。
> - 每列的元素从上到下升序排列。
```c++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix[0].size();
        int i = 0, j = n - 1;
        while (i < m && j >= 0) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] < target) {
                i++;
            } else {
                j--;
            }
        }
        return false;
    }
};
```

## 链表

[160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)
```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode *node1 = headA;
        ListNode *node2 = headB;
        while (node1 != node2) {
            node1 = node1 ? node1->next : headB;
            node2 = node2 ? node2->next : headA;
        }
        return node1;
    }
};
```

[206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)
```c++
class Solution {
public:
    // 迭代
    ListNode* reverseList(ListNode* head) {
        ListNode *prev = nullptr;
        ListNode *node = head;
        while (node) {
            ListNode *next = node->next;
            node->next = prev;
            prev = node;
            node = next;
        }
        return prev;
    }
    // 递归
    ListNode* reverseList(ListNode* head) {
        if (head == nullptr || head->next == nullptr) {
            return head;
        }
        ListNode *node = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        return node;
    }
};
```

[234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)
> 用 O(n) 时间复杂度和 O(1) 空间复杂度解决此题
```c++
class Solution {
    ListNode *reverseListNode(ListNode *head) {
        ListNode *prev = nullptr;
        ListNode *node = head;
        while (node) {
            ListNode *next = node->next;
            node->next = prev;
            prev = node;
            node = next;
        }
        return prev;
    }

public:
    bool isPalindrome(ListNode* head) {
        int num = 0;
        for (auto p = head; p; p = p->next) {
            num++;
        }
        if (num <= 1) {
            return true;
        }
        int half = num / 2;
        ListNode *a = head;
        while (half--) {
            a = a->next;
        }
        a = reverseListNode(a);
        ListNode *b = head;
        while (a && b) {
            if (a->val != b->val) {
                return false;
            }
            a = a->next;
            b = b->next;
        }
        return true;
    }
};
```

[141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)
```c++
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if (head == nullptr || head->next == nullptr) {
            return false;
        }
        ListNode *fast = head;
        ListNode *slow = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) {
                return true;
            }
        }
        return false;
    }
};
```

[142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)
```c++
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        if (head == nullptr || head->next == nullptr) {
            return nullptr;
        }
        ListNode *slow = head;
        ListNode *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) {
                break;
            }
        }
        if (fast == nullptr || fast->next == nullptr) {
            return nullptr;
        }

        slow = head;
        while (slow != fast) {
            slow = slow->next;
            fast = fast->next;
        }
        return slow;
    }
};
```

[21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)
```c++
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode *dummpy = new ListNode(0);
        ListNode *prev = dummpy;
        while (list1 && list2) {
            if (list1->val <= list2->val) {
                prev->next = list1;
                prev = prev->next;
                list1 = list1->next;
            } else {
                prev->next = list2;
                prev = prev->next;
                list2 = list2->next;
            }
        }
        if (list1) {
            prev->next = list1;
        }
        if (list2) {
            prev->next = list2;
        }
        return dummpy->next;
    }
};
```

[2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)
> 输入：l1 = [2,4,3], l2 = [5,6,4]  
> 输出：[7,0,8]  
> 解释：342 + 465 = 807.
```c++
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *dummpy = new ListNode(0);
        ListNode *prev = dummpy;
        int c = 0;
        while (l1 || l2 || c) {
            if (l1) {
                c += l1->val;
                l1 = l1->next;
            }
            if (l2) {
                c += l2->val;
                l2 = l2->next;
            }
            prev->next = new ListNode(c % 10);
            prev = prev->next;
            c /= 10;
        }
        return dummpy->next;
    }
};
```

[19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)
> 输入：head = [1,2,3,4,5], n = 2  
> 输出：[1,2,3,5]
```c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode *dummpy = new ListNode(0, head);
        ListNode *low = head;
        ListNode *fast = head;

        for (int i = 0; i < n && fast; i++) {
            fast = fast->next;
        }

        ListNode *prev = dummpy;

        while (fast) {
            low = low->next;
            fast = fast->next;
            prev = prev->next;
        }

        prev->next = low->next;
        return dummpy->next;
    }
};
```

[24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)
```c++
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode *dummpy = new ListNode(0, head);
        ListNode *prev = dummpy;
        while (prev->next && prev->next->next) {
            ListNode *a = prev->next;
            ListNode *b = prev->next->next;
            prev->next = b;
            a->next = b->next;
            b->next = a;

            prev = a;
        }
        return dummpy->next;
    }
};
```

[25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)
> 输入：head = [1,2,3,4,5], k = 3
> 输出：[3,2,1,4,5]
```c++
class Solution {
    ListNode *reverse(ListNode *head) {
        ListNode *prev = nullptr;
        ListNode *node = head;
        while (node) {
            ListNode *next = node->next;
            node->next = prev;
            prev = node;
            node = next;
        }
        return prev;
    }

    ListNode *getK(ListNode *node, int k) {
        for (int i = 0; i < k - 1 && node; i++) {
            node = node->next;
        }
        return node;
    }

public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode *dummpy = new ListNode(0, head);
        ListNode *prev = dummpy;
        ListNode *tail = getK(head, k);
        while (tail) {
            ListNode *nextHead = tail->next;
            tail->next = nullptr;
            reverse(head);
            prev->next = tail;
            head->next = nextHead;

            prev = head;
            head = nextHead;
            tail = getK(head, k);
        }
        return dummpy->next;
    }
};
```

[138. 随机链表的复制](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)
```c++
class Solution {
    unordered_map<Node*, Node*> table;

public:
    Node* copyRandomList(Node* head) {
        if (head == nullptr) {
            return nullptr;
        }

        if (table.find(head) == table.end()) {
            Node *newHead = new Node(head->val);
            table[head] = newHead;
            newHead->next = copyRandomList(head->next);
            newHead->random = copyRandomList(head->random);
        }
        return table[head];
    }
};
```

[148. 排序链表](https://leetcode-cn.com/problems/sort-list/)
> 在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序
```c++
class Solution {
    ListNode *merge(ListNode *head1, ListNode *head2) {
        ListNode *dummpy = new ListNode(0);
        ListNode *pre = dummpy;
        ListNode *node1 = head1;
        ListNode *node2 = head2;
        while (node1 && node2) {
            if (node1->val <= node2->val) {
                pre->next = node1;
                pre = pre->next;
                node1 = node1->next;
            } else {
                pre->next = node2;
                pre = pre->next;
                node2 = node2->next;
            }
        }
        if (node1) {
            pre->next = node1;
        }
        if (node2) {
            pre->next = node2;
        }
        return dummpy->next;
    }

public:
    ListNode* sortList(ListNode* head) {
        int length = 0;
        for (ListNode *p = head; p; p = p->next) {
            length++;
        }
        ListNode *dummpy = new ListNode(0, head);
        for (int subLength = 1; subLength < length; subLength += subLength) {
            ListNode *pre = dummpy;
            ListNode *cur = dummpy->next;
            while (cur) {
                ListNode *head1 = cur;
                for (int i = 1; i < subLength && cur->next; i++) {
                    cur = cur->next;
                }
                ListNode *head2 = cur->next;
                cur->next = nullptr;
                cur = head2;
                for (int i = 1; i < subLength && cur && cur->next; i++) {
                    cur = cur->next;
                }
                ListNode *next = nullptr;
                if (cur) {
                    next = cur->next;
                    cur->next = nullptr;
                }
                ListNode *merged = merge(head1, head2);
                pre->next = merged;
                while (pre->next) {
                    pre = pre->next;
                }
                cur = next;
            }
        }
        return dummpy->next;
    }
};
```

[23. 合并 K 个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)
```c++
class Solution {
    ListNode *merge(ListNode *head1, ListNode *head2) {
        ListNode *dummpy = new ListNode(0);
        ListNode *pre = dummpy;
        ListNode *node1 = head1;
        ListNode *node2 = head2;
        while (node1 && node2) {
            if (node1->val <= node2->val) {
                pre->next = node1;
                pre = pre->next;
                node1 = node1->next;
            } else {
                pre->next = node2;
                pre = pre->next;
                node2 = node2->next;
            }
        }
        if (node1) {
            pre->next = node1;
        }
        if (node2) {
            pre->next = node2;
        }
        return dummpy->next;
    }

    ListNode *mergeKListsHelper(vector<ListNode*> &lists, int left, int right) {
        if (left > right) {
            return nullptr;
        }
        if (left == right) {
            return lists[left];
        }
        int mid = left + right >> 1;
        return merge(mergeKListsHelper(lists, left, mid), mergeKListsHelper(lists, mid + 1, right));
    }

public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        return mergeKListsHelper(lists, 0, lists.size() - 1);
    }
};
```

[146. LRU 缓存](https://leetcode-cn.com/problems/lru-cache/)
```c++
class LRUCache {
    class Node {
    public:
        int key, value;
        Node *prev, *next;
        Node(int key, int value) : key(key), value(value), prev(nullptr), next(nullptr) {}
    };
    Node *head, *tail;
    unordered_map<int, Node*> table;
    int n;

    void remove(Node *node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void insert(Node *node) {
        node->prev = tail->prev;
        node->next = tail;
        node->prev->next = node;
        tail->prev = node;
    }

public:
    LRUCache(int capacity) {
        n = capacity;
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head->next = tail;
        tail->prev = head;
    }
    
    int get(int key) {
        if (table.find(key) == table.end()) {
            return -1;
        }
        Node *node = table[key];
        remove(node);
        insert(node);
        return node->value;
    }
    
    void put(int key, int value) {
        if (table.find(key) == table.end()) {
            if (n == table.size()) {
                Node *first = head->next;
                remove(first);
                table.erase(first->key);
            }
            Node *node = new Node(key, value);
            insert(node);
            table[key] = node;
        } else {
            Node *node = table[key];
            node->value = value;
            remove(node);
            insert(node);
        }
    }
};
```

## 二叉树

[94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)
```c++
class Solution {
    vector<int> ans;

public:
    vector<int> inorderTraversal(TreeNode* root) {
        if (root == nullptr) {
            return ans;
        }
        inorderTraversal(root->left);
        ans.emplace_back(root->val);
        inorderTraversal(root->right);
        return ans;
    }
};
```

[104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)
```c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr) {
            return 0;
        }
        int left = maxDepth(root->left);
        int right = maxDepth(root->right);
        return max(left, right) + 1;
    }
};
```

[226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)
```c++
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (root == nullptr) {
            return root;
        }
        TreeNode *left = invertTree(root->left);
        TreeNode *right = invertTree(root->right);
        root->left = right;
        root->right = left;
        return root;
    }
};
```

[101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)
```c++
class Solution {
    bool isSymmetricHelper(TreeNode *left, TreeNode *right) {
        if (!left && !right) {
            return true;
        }
        if (!left || !right) {
            return false;
        }
        if (left->val != right->val) {
            return false;
        }
        return isSymmetricHelper(left->left, right->right) && isSymmetricHelper(left->right, right->left);
    }

public:
    bool isSymmetric(TreeNode* root) {
        return isSymmetricHelper(root, root);
    }
};
```

[543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)
> 二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。
```c++
class Solution {
    int ans;

    int depth(TreeNode *node) {
        if (node == nullptr) {
            return 0;
        }
        int left = depth(node->left);
        int right = depth(node->right);
        ans = max(ans, left + right);
        return max(left, right) + 1;
    }

public:
    int diameterOfBinaryTree(TreeNode* root) {
        ans = 0;
        depth(root);
        return ans;
    }
};
```

[102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)
```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        if (root == nullptr) {
            return ans;
        }
        queue<TreeNode*> q;
        q.push(root);
        while (q.size()) {
            int sz = q.size();
            vector<int> path;
            while (sz--) {
                TreeNode *front = q.front();
                q.pop();
                path.emplace_back(front->val);
                if (front->left) {
                    q.push(front->left);
                }
                if (front->right) {
                    q.push(front->right);
                }
            }
            ans.emplace_back(path);
        }
        return ans;
    }
};
```

[108. 将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/)
```c++
class Solution {
    TreeNode *buildTree(vector<int> &nums, int left, int right) {
        if (left > right) {
            return nullptr;
        }
        int mid = left + right >> 1;
        TreeNode *node = new TreeNode(nums[mid]);
        node->left = buildTree(nums, left, mid - 1);
        node->right = buildTree(nums, mid + 1, right);
        return node;
    }

public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return buildTree(nums, 0, nums.size() - 1);
    }
};
```

[98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)
```c++
class Solution {
    using LL = long long;

    bool isValidBSTHelper(TreeNode *node, LL lower, LL upper) {
        if (node == nullptr) {
            return true;
        }

        if (node->val <= lower || node->val >= upper) {
            return false;
        }

        return isValidBSTHelper(node->left, lower, node->val) && isValidBSTHelper(node->right, node->val, upper);
    }

public:
    bool isValidBST(TreeNode* root) {
        return isValidBSTHelper(root, LONG_LONG_MIN, LONG_LONG_MAX);
    }
};
```

[230. 二叉搜索树中第 K 小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)
> 给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 小的元素（从 1 开始计数）。
```c++
class Solution {
    int countNum(TreeNode *node) {
        if (node == nullptr) {
            return 0;
        }
        int left = countNum(node->left);
        int right = countNum(node->right);
        return left + right + 1;
    }

public:
    int kthSmallest(TreeNode* root, int k) {
        int leftNum = countNum(root->left);
        if (leftNum >= k) {
            return kthSmallest(root->left, k);
        } else if (leftNum + 1 == k) {
            return root->val;
        } else {
            return kthSmallest(root->right, k - leftNum - 1);
        }
    }
};
```

[199. 二叉树的右视图](https://leetcode-cn.com/problems/binary-tree-right-side-view/)
```c++
class Solution {
    vector<int> ans;

    void levelOrder(TreeNode *node, int level) {
        if (node == nullptr) {
            return ;
        }

        if (level >= ans.size()) {
            ans.emplace_back(node->val);
        }
        levelOrder(node->right, level + 1);
        levelOrder(node->left, level + 1);
    }

public:
    vector<int> rightSideView(TreeNode* root) {
        levelOrder(root, 0);
        return ans;
    }
};
```

[114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)
> 给你二叉树的根结点 root ，请你将它展开为一个单链表：
> - 展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
> - 展开后的单链表应该与二叉树 先序遍历 顺序相同。
```c++
class Solution {
public:
    void flatten(TreeNode* root) {
        while (root) {
            TreeNode *node = root->left;
            if (node) {
                while (node->right) {
                    node = node->right;
                }
                node->right = root->right;
                root->right = root->left;
                root->left = nullptr;
            }
            root = root->right;
        }
    }
};
```

[105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
```c++
class Solution {
    unordered_map<int, int> table;

    TreeNode *build(vector<int> &preorder, vector<int> &inorder, int pl, int pr, int il, int ir) {
        if (pl > pr) {
            return nullptr;
        }
        TreeNode *node = new TreeNode(preorder[pl]);
        int index = table[preorder[pl]];
        node->left = build(preorder, inorder, pl + 1, pl + 1 + index - 1 - il + 1 - 1, il, index - 1);
        node->right = build(preorder, inorder, pl + 1 + index - 1 - il + 1 - 1 + 1, pr, index + 1, ir);
        return node;
    }

public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        for (int i = 0; i < inorder.size(); i++) {
            table[inorder[i]] = i;
        }
        return build(preorder, inorder, 0, preorder.size() - 1, 0, inorder.size() - 1);
    }
};
```

[437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)
> 给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。  
> 路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
```c++
class Solution {
    using LL = long long;
    unordered_map<LL, LL> table;
    LL ans;

    void dfs(TreeNode *node, LL targetSum, LL sum) {
        if (node == nullptr) {
            return ;
        }
        sum += node->val;
        ans += table[sum - targetSum];
        table[sum]++;
        dfs(node->left, targetSum, sum);
        dfs(node->right, targetSum, sum);
        table[sum]--;
    }

public:
    int pathSum(TreeNode* root, int targetSum) {
        ans = 0;
        table[0] = 1;
        dfs(root, targetSum, 0);
        return ans;
    }
};
```

[236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)
```c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (root == nullptr) {
            return nullptr;
        }
        if (root == p || root == q) {
            return root;
        }
        TreeNode *left = lowestCommonAncestor(root->left, p, q);
        TreeNode *right = lowestCommonAncestor(root->right, p, q);
        if (left && right) {
            return root;
        }
        return left ? left : right;
    }
};
```

[124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)
> 输入：root = [-10,9,20,null,null,15,7]  
> 输出：42  
> 解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42
```c++
class Solution {
    int ans;

    int dfs(TreeNode *node) {
        if (node == nullptr) {
            return 0;
        }
        int left = max(0, dfs(node->left));
        int right = max(0, dfs(node->right));
        ans = max(ans, left + right + node->val);
        return node->val + max(left, right);
    }

public:
    int maxPathSum(TreeNode* root) {
        ans = INT_MIN;
        dfs(root);
        return ans;
    }
};
```

[572. 另一棵树的子树](https://leetcode-cn.com/problems/subtree-of-another-tree/)
```c++
class Solution {
    vector<int> s{0};
    vector<int> p{0};

    void preOrder(TreeNode *node, vector<int> &vec) {
        if (node == nullptr) {
            vec.emplace_back(-1e4 - 1);
            return ;
        }
        vec.emplace_back(node->val);
        preOrder(node->left, vec);
        preOrder(node->right, vec);
    }

public:
    bool isSubtree(TreeNode* root, TreeNode* subRoot) {
        preOrder(root, s);
        preOrder(subRoot, p);
        int ans = -1;
        int n = s.size() - 1, m = p.size() - 1;
        vector<int> next(m + 1, 0);
        for (int i = 2, j = 0; i <= m; i++) {
            while (j && p[i] != p[j + 1]) {
                j = next[j];
            }
            if (p[i] == p[j + 1]) {
                j++;
            }
            next[i] = j;
        }
        for (int i = 1, j = 0; i <= n; i++) {
            while (j && s[i] != p[j + 1]) {
                j = next[j];
            }
            if (s[i] == p[j + 1]) {
                j++;
            }
            if (j == m) {
                return true;
            }
        }
        return false;
    }
};
```

## 图论

[200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)
> 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
```c++
class Solution {
    vector<int> dx{-1, 0, 1, 0};
    vector<int> dy{0, 1, 0, -1};

    void dfs(vector<vector<char>> &grid, int x, int y) {
        grid[x][y] = '0';
        for (int i = 0; i < dx.size(); i++) {
            int a = x + dx[i];
            int b = y + dy[i];
            if (a < 0 || a >= grid.size() || b < 0 || b >= grid[0].size() || grid[a][b] == '0') {
                continue;
            }
            dfs(grid, a, b);
        }
    }

public:
    int numIslands(vector<vector<char>>& grid) {
        int ans = 0;
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == '1') {
                    ans++;
                    dfs(grid, i, j);
                }
            }
        }
        return ans;
    }
};
```

[994. 腐烂的橘子](https://leetcode-cn.com/problems/rotting-oranges/)
> 在给定的网格中，每个单元格可以有以下三个值之一：
> - 值 0 代表空单元格；
> - 值 1 代表新鲜橘子；
> - 值 2 代表腐烂的橘子。
> 
> 每分钟，任何与腐烂橘子（在 4 个正方向上）相邻的新鲜橘子都会腐烂。  
> 返回直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1。
```c++
class Solution {
    using PII = pair<int, int>;
    vector<int> dx = {-1, 0, 1, 0};
    vector<int> dy = {0, 1, 0, -1};

public:
    int orangesRotting(vector<vector<int>>& grid) {
        queue<PII> q;
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == 2) {
                    q.push({i, j});
                }
            }
        }
        int ans = 0;
        if (q.size()) {
            ans--;
        }
        while (q.size()) {
            ans++;
            int sz = q.size();
            while (sz--) {
                auto front = q.front();
                q.pop();
                int x = front.first;
                int y = front.second;
                for (int i = 0; i < dx.size(); i++) {
                    int a = x + dx[i];
                    int b = y + dy[i];
                    if (a < 0 || a >= grid.size() || b < 0 || b >= grid[0].size() || grid[a][b] != 1) {
                        continue;
                    }
                    grid[a][b] = 2;
                    q.push({a, b});
                }
            }
        }
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == 1) {
                    return -1;
                }
            }
        }
        return ans;
    }
};
```

[207. 课程表](https://leetcode-cn.com/problems/course-schedule/)
> 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。  
> 在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。  
> - 例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
>
> 请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。
```c++
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> edge(numCourses);
        vector<int> indegree(numCourses);
        for (const auto &e : prerequisites) {
            int a = e[0], b = e[1];
            edge[b].emplace_back(a);
            indegree[a]++;
        }
        queue<int> q;
        for (int i = 0; i < indegree.size(); i++) {
            if (indegree[i] == 0) {
                q.push(i);
            } 
        }
        int cnt = 0;
        while (q.size()) {
            cnt++;
            int front = q.front();
            q.pop();
            for (const auto &node : edge[front]) {
                indegree[node]--;
                if (indegree[node] == 0) {
                    q.push(node);
                }
            }
        }
        return cnt == numCourses;
    }
};
```

[208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)
```c++
// Trie trie = new Trie();
// trie.insert("apple");
// trie.search("apple");   // 返回 True
// trie.search("app");     // 返回 False
// trie.startsWith("app"); // 返回 True
// trie.insert("app");
// trie.search("app");     // 返回 True
class Trie {
    class Node {
    public:
        bool isEnd;
        vector<Node*> son;
        Node() : isEnd(false), son(26, nullptr) {}
    };
    Node *root;

public:
    Trie() {
        root = new Node();
    }
    
    void insert(string word) {
        Node *node = root;
        for (const auto &ch : word) {
            int index = ch - 'a';
            if (!node->son[index]) {
                node->son[index] = new Node();
            }
            node = node->son[index];
        }
        node->isEnd = true;
    }
    
    bool search(string word) {
        Node *node = root;
        for (const auto &ch : word) {
            int index = ch - 'a';
            if (!node->son[index]) {
                return false;
            }
            node = node->son[index];
        }
        return node->isEnd;
    }
    
    bool startsWith(string prefix) {
        Node *node = root;
        for (const auto &ch : prefix) {
            int index = ch - 'a';
            if (!node->son[index]) {
                return false;
            }
            node = node->son[index];
        }
        return true;
    }
};
```

## 回溯

[46. 全排列](https://leetcode-cn.com/problems/permutations/)
```c++
class Solution {
    vector<vector<int>> ans;
    vector<int> path;
    vector<bool> used;

    void dfs(vector<int> &nums, int cur) {
        if (cur == nums.size()) {
            ans.emplace_back(path);
            return ;
        }

        for (int i = 0; i < nums.size(); i++) {
            if (!used[i]) {
                used[i] = true;
                path.emplace_back(nums[i]);
                dfs(nums, cur + 1);
                path.pop_back();
                used[i] = false;
            }
        }
    }

public:
    vector<vector<int>> permute(vector<int>& nums) {
        used = vector<bool>(nums.size());
        dfs(nums, 0);
        return ans;
    }
};
```

[78. 子集](https://leetcode-cn.com/problems/subsets/)
> 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。  
> 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。  
> 输入：nums = [1,2,3]  
> 输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```c++
class Solution {
    vector<vector<int>> ans;
    vector<int> path;

    void dfs(vector<int> &nums, int cur) {
        if (cur == nums.size()) {
            ans.emplace_back(path);
            return ;
        }
        path.emplace_back(nums[cur]);
        dfs(nums, cur + 1);
        path.pop_back();
        dfs(nums, cur + 1);
    }

public:
    vector<vector<int>> subsets(vector<int>& nums) {
        dfs(nums, 0);
        return ans;
    }
};
```

[17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)
> 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
```c++
class Solution {
    unordered_map<char, string> table = {
        {'2', "abc"},
        {'3', "def"},
        {'4', "ghi"},
        {'5', "jkl"},
        {'6', "mno"},
        {'7', "pqrs"},
        {'8', "tuv"},
        {'9', "wxyz"},
    };
    vector<string> ans;

    void dfs(const string &digits, int cur, string path) {
        if (cur == digits.size()) {
            ans.emplace_back(path);
            return;
        }
        for (const auto &ch : table[digits[cur]]) {
            dfs(digits, cur + 1, path + ch);
        }
    }

public:
    vector<string> letterCombinations(string digits) {
        if (digits == "") {
            return ans;
        }
        dfs(digits, 0, "");
        return ans;
    }
};
```

[39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)
> 输入：candidates = [2,3,6,7], target = 7  
> 输出：[[2,2,3],[7]]  
> 这是完全背包问题，但是要求输出所有解，所以需要使用回溯算法
```c++
class Solution {
    vector<vector<int>> ans;
    vector<int> path;

    void dfs(vector<int> &candidates, int cur, int target) {
        if (target < 0) {
            return ;
        }
        if (target == 0) {
            ans.emplace_back(path);
            return;
        }
        if (cur == candidates.size()) {
            return;
        }
        path.emplace_back(candidates[cur]);
        dfs(candidates, cur, target - candidates[cur]);
        path.pop_back();
        dfs(candidates, cur + 1, target);
    }

public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        dfs(candidates, 0, target);
        return ans;
    }
};
```

[22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)
> 输入：n = 3  
> 输出：["((()))","(()())","(())()","()(())","()()()"]
```c++
class Solution {
    vector<string> ans;

    void dfs(int n, int left, int right, string path) {
        if (left == n && right == n) {
            ans.emplace_back(path);
            return;
        }
        if (left < n) {
            dfs(n, left + 1, right, path + "(");
        }
        if (right < n && left > right) {
            dfs(n, left, right + 1, path + ")");
        }
    }

public:
    vector<string> generateParenthesis(int n) {
        dfs(n, 0, 0, "");
        return ans;
    }
};
```

[79. 单词搜索](https://leetcode-cn.com/problems/word-search/)
> 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
```c++
class Solution {
    vector<int> dx = {-1, 0, 1, 0};
    vector<int> dy = {0, 1, 0, -1};

    bool dfs(vector<vector<char>> &board, const string &word, int cur, int x, int y) {
        if (board[x][y] != word[cur]) {
            return false;
        }
        if (cur == word.size() - 1) {
            return true;
        }
        char tmp = board[x][y];
        board[x][y] = '.';
        for (int i = 0; i < dx.size(); i++) {
            int a = x + dx[i];
            int b = y + dy[i];
            if (a < 0 || a >= board.size() || b < 0 || b >= board[0].size() || board[a][b] == '.') {
                continue;
            }
            if (dfs(board, word, cur + 1, a, b)) {
                return true;
            }
        }
        board[x][y] = tmp;
        return false;
    }
    
public:
    bool exist(vector<vector<char>>& board, string word) {
        for (int i = 0; i < board.size(); i++) {
            for (int j = 0; j < board[0].size(); j++) {
                if (dfs(board, word, 0, i, j)) {
                    return true;
                }
            }
        }
        return false;
    }
};
```

[131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/)
> 输入：s = "aab"  
> 输出：[["a","a","b"],["aa","b"]]
```c++
class Solution {
    vector<vector<bool>> f;
    vector<vector<string>> ans;
    vector<string> path;

    void dfs(const string &s, int cur) {
        if (cur == s.size()) {
            ans.emplace_back(path);
            return ;
        }
        for (int i = cur; i < s.size(); i++) {
            if (f[cur][i]) {
                path.emplace_back(s.substr(cur, i - cur + 1));
                dfs(s, i + 1);
                path.pop_back();
            }
        }
    }

public:
    vector<vector<string>> partition(string s) {
        int n = s.size();
        f = vector<vector<bool>>(n, vector<bool>(n));

        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                if (s[i] != s[j]) {
                    continue;
                }
                if (i == j || i + 1 == j) {
                    f[i][j] = true;
                } else {
                    f[i][j] = f[i + 1][j - 1];
                }
            }
        }
        dfs(s, 0);
        return ans;
    }
};
```

[51. N 皇后](https://leetcode-cn.com/problems/n-queens/)
```c++
class Solution {
    int n_;
    vector<bool> col, dg, udg;
    vector<vector<string>> ans;
    vector<string> path;

    void dfs(int cur) {
        if (cur == n_) {
            ans.emplace_back(path);
            return ;
        }

        for (int i = 0; i < n_; i++) {
            if (!col[i] && !dg[n_ + cur - i] && !udg[i + cur]) {
                col[i] = dg[n_ + cur - i] = udg[i + cur] = true; 
                path[cur][i] = 'Q';
                dfs(cur + 1);
                path[cur][i] = '.';
                col[i] = dg[n_ + cur - i] = udg[i + cur] = false; 
            }
        }
    }

public:
    vector<vector<string>> solveNQueens(int n) {
        n_ = n;
        col = vector<bool>(n);
        dg = udg = vector<bool>(2 * n);
        path = vector<string>(n, string(n, '.'));
        dfs(0);
        return ans;
    }
};
```

## 二分查找

[35. 搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/)
> 输入: nums = [1,3,5,6], target = 2  
> 输出: 1  
> 输入: nums = [1,3,5,6], target = 7  
> 输出: 4
```c++
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left < right) {
            int mid = left + right >> 1;
            if (nums[mid] >= target) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return target > nums[left] ? left + 1 : left;
    }
};
```

[74. 搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/)
> 给你一个满足下述两条属性的 m x n 整数矩阵：  
> - 每行中的整数从左到右按非严格递增顺序排列。  
> - 每行的第一个整数大于前一行的最后一个整数。  
> 
> 给你一个整数 target ，如果 target 在矩阵中，返回 true ；否则，返回 false 。
```c++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix[0].size();
        int left = 0, right = m * n - 1;
        while (left < right) {
            int mid = left + right >> 1;
            if (matrix[mid / n][mid % n] >= target) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return matrix[left / n][left % n] == target;
    }
};
```

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
```c++
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        if (nums.empty()) {
            return {-1, -1};
        }

        vector<int> ans;
        int left = 0, right = nums.size() - 1;
        while (left < right) {
            int mid = left + right >> 1;
            if (nums[mid] >= target) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        if (nums[left] != target) {
            return {-1, -1};
        }
        ans.emplace_back(left);

        left = 0, right = nums.size() - 1;
        while (left < right) {
            int mid = left + right + 1 >> 1;
            if (nums[mid] <= target) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        ans.emplace_back(left);
        return ans;
    }
};
```

[33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)
> 输入：nums = [4,5,6,7,0,1,2], target = 0  
> 输出：4
```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left < right) {
            int mid = left + right + 1 >> 1;
            if (nums[mid] > nums[0]) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        if (target >= nums[0]) {
            left = 0;
        } else {
            left = left + 1;
            right = nums.size() - 1;
        }
        if (left > right) {
            return - 1;
        }
        while (left < right) {
            int mid = left + right + 1 >> 1;
            if (nums[mid] <= target) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        return nums[left] == target ? left : -1;
    }
};
```

[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)
```c++
class Solution {
public:
    int findMin(vector<int>& nums) {
        int left = 0, right = nums.size() - 1;
        while (left < right) {
            int mid = left + right >> 1;
            if (nums[mid] < nums[0]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return min(nums[left], nums[0]);
    }
};
```

[4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)
> 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。算法的时间复杂度应该为 O(log (m+n)) 。
```c++
class Solution {
    int getKthElement(vector<int> &nums1, vector<int> &nums2, int k) {
        int m = nums1.size();
        int n = nums2.size();
        int index1 = 0, index2 = 0;

        while (true) {
            if (index1 == m) {
                return nums2[index2 + k];
            }
            if (index2 == n) {
                return nums1[index1 + k];
            }
            if (k == 0) {
                return min(nums1[index1], nums2[index2]);
            }

            int newIndex1 = min(index1 + (k + 1) / 2 - 1, m - 1);
            int newIndex2 = min(index2 + (k + 1) / 2 - 1, n - 1);
            if (nums1[newIndex1] <= nums2[newIndex2]) {
                k -= newIndex1 - index1 + 1;
                index1 = newIndex1 + 1;
            } else {
                k -= newIndex2 - index2 + 1;
                index2 = newIndex2 + 1;
            }
        }
    }
    
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int totalLength = nums1.size() + nums2.size();
        if (totalLength % 2) {
            return getKthElement(nums1, nums2, totalLength / 2);
        } else {
            int left = getKthElement(nums1, nums2, totalLength / 2 - 1);
            int right = getKthElement(nums1, nums2, totalLength / 2);
            return (left + right) / 2.0;
        }
    }
};
```

## 栈

[20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)
> 输入：s = "()[]{}"  
> 输出：true
```c++
class Solution {
public:
    bool isValid(string s) {
        stack<char> stk;

        unordered_map<char, char> table = {
            {'(', ')'},
            {'{', '}'},
            {'[', ']'}
        };

        for (const auto &ch : s) {
            if (table.find(ch) != table.end()) {
                stk.push(ch);
            } else {
                if (!stk.empty() && table[stk.top()] == ch) {
                    stk.pop();
                } else {
                    return false;
                }
            }
        }
        return stk.empty();
    }
};
```

[155. 最小栈](https://leetcode-cn.com/problems/min-stack/)
```c++
// MinStack minStack = new MinStack();
// minStack.push(-2);
// minStack.push(0);
// minStack.push(-3);
// minStack.getMin();   --> 返回 -3.
// minStack.pop();
// minStack.top();      --> 返回 0.
// minStack.getMin();   --> 返回 -2.
class MinStack {
    stack<int> stk, f;
public:
    MinStack() {

    }
    
    void push(int val) {
        stk.push(val);
        if (f.empty() || val <= f.top()) {
            f.push(val);
        }
    }
    
    void pop() {
        if (stk.top() == f.top()) {
            f.pop();
        }
        stk.pop();
    }
    
    int top() {
        return stk.top();
    }
    
    int getMin() {
        return f.top();
    }
};
```

[394. 字符串解码](https://leetcode-cn.com/problems/decode-string/)
> 输入：s = "3[a]2[bc]"  
> 输出："aaabcbc"
```c++
class Solution {
    string dfs(const string &s, int &cur) {
        string ans;
        while (cur < s.size() && s[cur] != ']') {
            if (s[cur] >= 'a' && s[cur] <= 'z') {
                ans += s[cur++];
            } else {
                int k = cur;
                while (s[k] >= '0' && s[k] <= '9') k++;
                int num = stoi(s.substr(cur, k - cur));
                cur = k + 1;
                string sub = dfs(s, cur);
                cur++;
                while (num--) {
                    ans += sub;
                }
            }
        }
        return ans;
    }

public:
    string decodeString(string s) {
        int cur = 0;
        return dfs(s, cur);
    }
};
```

[739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)
> 给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。  
> 输入: temperatures = [73,74,75,71,69,72,76,73]  
> 输出: [1,1,4,2,1,1,0,0]
```c++
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        stack<int> stk;
        vector<int> ans(temperatures.size());
        for (int i = temperatures.size() - 1; i >= 0; i--) {
            while (!stk.empty() && temperatures[stk.top()] <= temperatures[i]) {
                stk.pop();
            }
            if (stk.empty()) {
                ans[i] = 0;
            } else {
                ans[i] = stk.top() - i;
            }
            stk.push(i);
        }
        return ans;
    }
};
```

[84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)
> 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。  
> 求在该柱状图中，能够勾勒出来的矩形的最大面积。
```c++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        int n = heights.size();
        vector<int> left(n), right(n);
        stack<int> stk;
        for (int i = 0; i < heights.size(); i++) {
            while (!stk.empty() && heights[stk.top()] >= heights[i]) {
                stk.pop();
            }
            if (stk.empty()) {
                left[i] = -1;
            } else {
                left[i] = stk.top();
            }
            stk.push(i);
        }

        stk = stack<int>();
        for (int i = heights.size() - 1; i >= 0; i--) {
            while (!stk.empty() && heights[stk.top()] >= heights[i]) {
                stk.pop();
            }
            if (stk.empty()) {
                right[i] = n;
            } else {
                right[i] = stk.top();
            }
            stk.push(i);
        }

        int ans = 0;
        for (int i = 0; i < heights.size(); i++) {
            ans = max(ans, (right[i] - left[i] - 1) * heights[i]);
        }
        return ans;
    }
};
```

## 堆

[215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)
> 给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。  
> 输入: [3,2,1,5,6,4], k = 2  
> 输出: 5
```c++
class Solution {
    int quick_sort(vector<int> &nums, int left, int right, int k) {
        if (left >= right) {
            return nums[k];
        }

        int i = left - 1, j = right + 1, x = nums[left + right >> 1];
        while (i < j) {
            do i++; while (nums[i] < x);
            do j--; while (nums[j] > x);
            if (i < j) swap(nums[i], nums[j]);
        }
        if (k <= j) return quick_sort(nums, left, j, k);
        else return quick_sort(nums, j + 1, right, k);
    }

public:
    int findKthLargest(vector<int>& nums, int k) {
        return quick_sort(nums, 0, nums.size() - 1, nums.size() - k);
    }
};
```

[347. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)
> 输入: nums = [1,1,1,2,2,3], k = 2  
> 输出: [1,2]
```c++
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> cnt;
        for (const auto &num : nums) {
            cnt[num]++;
        }

        vector<int> s(nums.size() + 1);
        for (const auto &item : cnt) {
            s[item.second]++;
        }

        int i = nums.size(), sum = 0;
        for (; i >= 0; i--) {
            sum += s[i];
            if (sum > k) {
                break;
            }
        }

        vector<int> ans;
        for (const auto &item : cnt) {
            if (item.second > i) {
                ans.emplace_back(item.first);
            }
        }
        return ans;
    }
};
```

[295. 数据流的中位数](https://leetcode-cn.com/problems/find-median-from-data-stream/)
```c++
class MedianFinder {
    priority_queue<int, vector<int>, greater<int>> up;
    priority_queue<int> down;

public:
    MedianFinder() {}
    
    void addNum(int num) {
        if (down.empty() || num <= down.top()) {
            down.push(num);
            if (down.size() > up.size() + 1) {
                up.push(down.top());
                down.pop();
            }
        } else {
            up.push(num);
            if (up.size() > down.size()) {
                down.push(up.top());
                up.pop();
            }
        }
    }
    
    double findMedian() {
        if ((up.size() + down.size()) % 2) {
            return down.top();
        } else {
            return (up.top() + down.top()) / 2.0;
        }
    }
};
```

## 贪心算法

[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)
> 输入：[7,1,5,3,6,4]  
> 输出：5  
> 解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int ans = 0;
        for (int i = 0, minPrice = INT_MAX; i < prices.size(); i++) {
            ans = max(ans, prices[i] - minPrice);
            minPrice = min(minPrice, prices[i]);
        }
        return ans;
    }
};
```

[55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)
> 输入：nums = [2,3,1,1,4]  
> 输出：true  
> 解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
```c++
class Solution {
public:
    bool canJump(vector<int>& nums) {
        for (int i = 0, j = 0; i < nums.size(); i++) {
            if (i > j) {
                return false;
            }
            j = max(j, nums[i] + i);
        }
        return true;
    }
};
```

[45. 跳跃游戏 II](https://leetcode-cn.com/problems/jump-game-ii/)
> 输入: nums = [2,3,1,1,4]  
> 输出: 2  
> 解释: 跳到最后一个位置的最小跳跃数是 2。从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
```c++
class Solution {
public:
    int jump(vector<int>& nums) {
        vector<int> f(nums.size());
        for (int i = 1, j = 0; i < nums.size(); i++) {
            while (j + nums[j] < i) {
                j++;
            }
            f[i] = f[j] + 1;
        }
        return f[nums.size() - 1];
    }
};
```

[763. 划分字母区间](https://leetcode-cn.com/problems/partition-labels/)
> 给你一个字符串 s 。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。  
> 注意，划分结果需要满足：将所有划分结果按顺序连接，得到的字符串仍然是 s 。  
> 返回一个表示每个字符串片段的长度的列表。
> 输入：s = "ababcbacadefegdehijhklij"  
> 输出：[9,7,8]  
> 解释：划分结果为 "ababcbaca"、"defegde"、"hijhklij" 。每个字母最多出现在一个片段中。像 "ababcbacadefegde", "hijhklij" 这样的划分是错误的，因为划分的片段数较少。 
```c++
class Solution {
public:
    vector<int> partitionLabels(string s) {
        unordered_map<char, int> last;
        for (int i = 0; i < s.size(); i++) {
            last[s[i]] = i;
        }
        vector<int> ans;
        int start = 0, end = 0;
        for (int i = 0; i < s.size(); i++) {
            end = max(end, last[s[i]]);
            if (i == end) {
                ans.emplace_back(end - start + 1);
                end = start = i + 1;
            }
        }
        return ans;
    }
};
```

## 动态规划

[70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/) 
```c++
class Solution {
    using LL = long long;

public:
    int climbStairs(int n) {
        LL a = 1, b = 1;
        while (n--) {
            int c = a + b;
            a = b;
            b = c;
        }
        return a;
    }
};
```

[118. 杨辉三角](https://leetcode-cn.com/problems/pascals-triangle/)
```c++
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> ans;
        for (int i = 0; i < numRows; i++) {
            vector<int> line(i + 1);
            line[0] = line[i] = 1;
            for (int j = 1; j < i; j++) {
                line[j] = ans[i - 1][j - 1] + ans[i - 1][j];
            }
            ans.emplace_back(line);
        }
        return ans;
    }
};
```

[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)
> 输入：[1,2,3,1]
> 输出：4
> 解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。偷窃到的最高金额 = 1 + 3 = 4 。
```c++
class Solution {
public:
    int rob(vector<int>& nums) {
        if (nums.size() == 1) {
            return nums[0];
        }
        
        vector<int> f(nums.size());
        f[0] = nums[0];
        f[1] = max(nums[0], nums[1]);
        for (int i = 2; i < nums.size(); i++) {
            f[i] = max(f[i - 2] + nums[i], f[i - 1]);
        }
        return f[nums.size() - 1];
    }
};
```

[279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)
> 给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。  
> 完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。  
> 输入：n = 13  
> 输出：2  
> 解释：13 = 4 + 9
```c++
class Solution {
public:
    int numSquares(int n) {
        vector<int> f(n + 1, n);
        for (int i = 1; i <= sqrt(n); i++) {
            f[i * i] = 1;
        }
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= sqrt(i); j++) {
                f[i] = min(f[i], f[i - j * j] + 1);
            }
        }
        return f[n];
    }
};
```

[322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)
> 输入：coins = [1, 2, 5], amount = 11  
> 输出：3  
> 解释：11 = 5 + 5 + 1
```c++
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        vector<int> f(amount + 1, INT_MAX - 1);
        f[0] = 0;
        for (const auto &v : coins) {
            for (int j = v; j <= amount; j++) {
                f[j] = min(f[j], f[j - v] + 1);
            }
        }
        return f[amount] == INT_MAX - 1 ? -1 : f[amount];
    }
};
```

[139. 单词拆分](https://leetcode-cn.com/problems/word-break/)
> 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 s 则返回 true。  
> 注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
```c++
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
        int n = s.size();
        vector<int> f(n + 1);
        f[0] = true;
        s = " " + s;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                if (f[j - 1] && wordSet.find(s.substr(j, i - j + 1)) != wordSet.end()) {
                    f[i] = true;
                }
            }
        }
        return f[n];
    }
};
```

[300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)
> 输入：nums = [10,9,2,5,3,7,101,18]  
> 输出：4  
> 解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
```c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        vector<int> q;
        for (const auto &num : nums) {
            if (q.empty() || q.back() < num) {
                q.emplace_back(num);
            } else {
                if (num <= q[0]) {
                    q[0] = num;
                } else {
                    int left = 0, right = q.size() - 1;
                    while (left < right) {
                        int mid = (left + right + 1) >> 1;
                        if (q[mid] < num) {
                            left = mid;
                        } else {
                            right = mid - 1;
                        }
                    }
                    q[left + 1] = num;
                }
            }
        }
        return q.size();
    }
};
```

[152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)
> 输入: nums = [2,3,-2,4]  
> 输出: 6  
> 解释: 子数组 [2,3] 有最大乘积 6。
```c++
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        double ans = nums[0], f = nums[0], g = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            double a = nums[i], fa = f * a, ga = g * a;
            f = max(a, max(fa, ga));
            g = min(a, min(fa, ga));
            ans = max(ans, f);
        }
        return ans;
    }
};
```

[416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)
> 输入：nums = [1,5,11,5]  
> 输出：true  
> 解释：数组可以分割成 [1, 5, 5] 和 [11] 。
```c++
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        if (nums.size() <= 1) {
            return false;
        }
        int sum = 0;
        for (const auto &num : nums) {
            sum += num;
        }
        if (sum % 2) {
            return false;
        }
        sum /= 2;

        vector<int> f(sum + 1);
        f[0] = true;
        for (const auto &v : nums) {
            for (int j = sum; j >= v; j--) {
                f[j] |= f[j - v];
            }
        }
        return f[sum];
    }
};
```

[32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)
> 给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。  
> 输入：s = ")()())"  
> 输出：4  
> 解释：最长有效括号子串是 "()()"
```
class Solution {
public:
    int longestValidParentheses(string s) {
        int ans = 0;
        vector<int> f(s.size());
        for (int i = 1; i < s.size(); i++) {
            if (s[i] == ')') {
                if (s[i - 1] == '(') {
                    f[i] = (i - 2 >= 0 ? f[i - 2] : 0) + 2;
                } else if (i - f[i - 1] - 1 >= 0 && s[i - f[i - 1] - 1] == '(') {
                    f[i] = (i - f[i - 1] - 2 >= 0 ? f[i - f[i - 1] - 2] : 0) + f[i - 1] + 2;
                }
                ans = max(ans, f[i]);
            }
        }
        return ans;
    }
};
```

## 多维动态规划

[62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)
> [0, 0] -> [m - 1, n - 1] 路径数量
```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> f(m, vector<int>(n));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!i && !j) f[i][j] = 1;
                else {
                    if (i) f[i][j] += f[i - 1][j];
                    if (j) f[i][j] += f[i][j - 1];
                }
            }
        }
        return f[m - 1][n - 1];
    }
};
```

[64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)
> 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
```c++
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<int>> f(m, vector<int>(n, INT_MAX));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!i && !j) f[i][j] = grid[i][j];
                else {
                    if (i) f[i][j] = min(f[i][j], f[i - 1][j] + grid[i][j]);
                    if (j) f[i][j] = min(f[i][j], f[i][j - 1] + grid[i][j]);
                }
            }
        }
        return f[m - 1][n - 1];
    }
};
```

[5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)
```c++
class Solution {
};
class Solution {
public:
    // dp 方法，时间复杂度 O(n^2)，空间复杂度 O(n^2)
    string longestPalindrome(string s) {
        int n = s.size();
        vector<vector<int>> f(n, vector<int>(n));
        string ans;
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                if (s[i] != s[j]) {
                    continue;
                }
                if (i == j || i + 1 == j) {
                    f[i][j] = true;
                } else {
                    f[i][j] = f[i + 1][j - 1];
                }

                if (f[i][j] && j - i + 1 > ans.size()) {
                    ans = s.substr(i, j - i + 1);
                }
            }
        }
        return ans;
    }
    // 中心扩展，时间复杂度 O(n^2)，空间复杂度 O(1)
    string longestPalindrome(string s) {
        string ans;
        for (int i = 0; i < s.size(); i++) {
            int left = i - 1, right = i + 1;
            while (left >= 0 && right < s.size() && s[left] == s[right]) {
                left--;
                right++;
            }
            if (ans.size() < right - left - 1) {
                ans = s.substr(left + 1, right - left - 1);
            }

            left = i, right = i + 1;
            while (left >= 0 && right < s.size() && s[left] == s[right]) {
                left--;
                right++;
            }
            if (ans.size() < right - left - 1) {
                ans = s.substr(left + 1, right - left - 1);
            }
        }
        return ans;
    }
};
```

[1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)
> 输入：text1 = "abcde", text2 = "ace"  
> 输出：3   
> 解释：最长公共子序列是 "ace" ，它的长度为 3 。

```c++
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.size(), n = text2.size();
        vector<vector<int>> f(m + 1, vector<int>(n + 1));
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                f[i][j] = max(f[i - 1][j], f[i][j - 1]);
                if (text1[i - 1] == text2[j - 1]) {
                    f[i][j] = max(f[i][j], f[i - 1][j - 1] + 1);
                }
            }
        }
        return f[m][n];
    }
};
```

[最长公共子串](https://www.nowcoder.com/practice/f33f5adc55f444baa0e0ca87ad8a6aac?tab=note)
```c++
class Solution {
public:
    string LCS(string str1, string str2) {
        int m = str1.size(), n = str2.size();
        vector<vector<int>> f(m + 1, vector<int>(n + 1));
        string ans;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str1[i - 1] != str2[j - 1]) {
                    f[i][j] = 0;
                } else {
                    f[i][j] = f[i - 1][j - 1] + 1;
                    if (f[i][j] > ans.size()) {
                        ans = str1.substr(i - f[i][j] + 1 - 1, f[i][j]);
                    }
                }
            }
        }
        return ans;
    }
};
```

[72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)
> 给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数 
> 你可以对一个单词进行如下三种操作：
> - 插入一个字符
> - 删除一个字符
> - 替换一个字符
```c++
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m = word1.size(), n = word2.size();
        vector<vector<int>> f(m + 1, vector<int>(n + 1));
        for (int i = 1; i <= m; i++) {
            f[i][0] = i;
        }
        for (int j = 1; j <= n; j++) {
            f[0][j] = j;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                f[i][j] = min(f[i - 1][j], f[i][j - 1]) + 1;
                int t = (word1[i - 1] != word2[j - 1]);
                f[i][j] = min(f[i][j], f[i - 1][j - 1] + t);
            }
        }
        return f[m][n];
    }
};
```

## 技巧

[136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)
> 给你一个 非空 整数数组 nums ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。  
> 你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。
```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ans = 0;
        for (const auto &num : nums) {
            ans ^= num;
        }
        return ans;
    }
};
```

[169. 多数元素](https://leetcode-cn.com/problems/majority-element/)
> 给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。  
> 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int candidate = 0, cnt = 0;
        for (const auto &num : nums) {
            if (cnt == 0) {
                candidate = num;
                cnt = 1;
            } else if (candidate != num) {
                cnt--;
            } else {
                cnt++;
            }
        }
        return candidate;
    }
};
```

[75. 颜色分类](https://leetcode-cn.com/problems/sort-colors/)
> 给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。  
> 我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
```c++
class Solution {
public:
    void sortColors(vector<int>& nums) {
        for (int left = 0, mid = 0, right = nums.size() - 1; mid <= right; ) {
            if (nums[mid] == 0) {
                swap(nums[left++], nums[mid++]);
            } else if (nums[mid] == 2) {
                swap(nums[mid], nums[right--]);
            } else {
                mid++;
            }
        }
    }
};
```

[31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)
> 必须 原地 修改，只允许使用额外常数空间。
```c++
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int k = nums.size() - 1;
        while (k - 1 >= 0 && nums[k - 1] >= nums[k]) {
            k--;
        }
        if (k == 0) {
            reverse(nums.begin() + k, nums.end());
            return ;
        }

        int j = nums.size() - 1;
        while (j >= k && nums[k - 1] >= nums[j]) {
            j--;
        }
        swap(nums[k - 1], nums[j]);
        reverse(nums.begin() + k, nums.end());
    }
};
```

[287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)
> 给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。  
```c++
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int slow = 0, fast = 0;
        while (true) {
            slow = nums[slow];
            fast = nums[nums[fast]];
            if (slow == fast) {
                break;
            }
        }
        slow = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }
};
```
