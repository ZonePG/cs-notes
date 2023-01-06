# 树与图的遍历

时间复杂度 O(n + m), n 表示点数，m 表示边数

## 深度优先遍历

**模板题**
- [AcWing 842. 排列数字](https://www.acwing.com/problem/content/844/)
- [AcWing 843. n-皇后问题](https://www.acwing.com/problem/content/845/)
- [AcWing 846. 树的重心](https://www.acwing.com/problem/content/848/)
```c++
int dfs(int u)
{
    st[u] = true; // st[u] 表示点u已经被遍历过

    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (!st[j]) dfs(j);
    }
}
```

## 宽度优先遍历

**模板题**
- [AcWing 844. 走迷宫](https://www.acwing.com/problem/content/846/)
- [AcWing 845. 八数码](https://www.acwing.com/problem/content/847/)
- [AcWing 847. 图中点的层次](https://www.acwing.com/problem/content/849/)
```c++
queue<int> q;
st[1] = true; // 表示1号点已经被遍历过
q.push(1);

while (q.size())
{
    int t = q.front();
    q.pop();

    for (int i = h[t]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (!st[j])
        {
            st[j] = true; // 表示点j已经被遍历过
            q.push(j);
        }
    }
}
```

## 拓扑排序

**模板题**
- [AcWing 848. 有向图的拓扑序列](https://www.acwing.com/problem/content/850/)
```c++
bool topsort()
{
    int hh = 0, tt = -1;

    // d[i] 存储点i的入度
    for (int i = 1; i <= n; i ++ )
        if (!d[i])
            q[ ++ tt] = i;

    while (hh <= tt)
    {
        int t = q[hh ++ ];

        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (-- d[j] == 0)
                q[ ++ tt] = j;
        }
    }

    // 如果所有点都入队了，说明存在拓扑序列；否则不存在拓扑序列。
    return tt == n - 1;
}
```