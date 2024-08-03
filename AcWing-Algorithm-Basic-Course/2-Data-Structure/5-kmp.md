# KMP

[AcWing 831. KMP字符串](https://www.acwing.com/problem/content/833/)
```c++
// s[]是长文本，p[]是模式串，n是s的长度，m是p的长度
求模式串的Next数组：
for (int i = 2, j = 0; i <= m; i ++ )
{
    while (j && p[i] != p[j + 1]) j = ne[j];
    if (p[i] == p[j + 1]) j ++ ;
    ne[i] = j;
}

// 匹配
for (int i = 1, j = 0; i <= n; i ++ )
{
    while (j && s[i] != p[j + 1]) j = ne[j];
    if (s[i] == p[j + 1]) j ++ ;
    if (j == m)
    {
        j = ne[j];
        // 匹配成功后的逻辑
    }
}
```

KMP：[leetcode 28. 找出字符串中第一个匹配项的下标](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/description/)
```c++
class Solution {
public:
    int strStr(string haystack, string needle) {
        int m = haystack.size(), n = needle.size();
        string s = " " + haystack, p = " " + needle;
        int ans = -1;
        vector<int> next(n + 1);
        for (int i = 2, j = 0; i <= n; i++) {
            while (j && p[i] != p[j + 1]) {
                j = next[j];
            }
            if (p[i] == p[j + 1]) {
                j++;
            }
            next[i] = j;
        }
        for (int i = 1, j = 0; i <= m; i++) {
            while (j && s[i] != p[j + 1]) {
                j = next[j];
            }
            if (s[i] == p[j + 1]) {
                j++;
            }
            if (j == n) {
                ans = i - j;
                break;
            } 
        }
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
