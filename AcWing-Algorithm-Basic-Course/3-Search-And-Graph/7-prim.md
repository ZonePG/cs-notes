# 朴素版 prim 算法

时间复杂度是 $O(n^2+m)$, n 表示点数，m 表示边数

**模板题**
- [AcWing 858. Prim算法求最小生成树](https://www.acwing.com/problem/content/860/)
```c++
int n;      // n表示点数
int g[N][N];        // 邻接矩阵，存储所有边
int dist[N];        // 存储其他点到当前最小生成树的距离
bool st[N];     // 存储每个点是否已经在生成树中


// 如果图不连通，则返回INF(值是0x3f3f3f3f), 否则返回最小生成树的树边权重之和
int prim()
{
    memset(dist, 0x3f, sizeof dist);

    int res = 0;
    for (int i = 0; i < n; i ++ )
    {
        int t = -1;
        for (int j = 1; j <= n; j ++ )
            if (!st[j] && (t == -1 || dist[t] > dist[j]))
                t = j;

        if (i && dist[t] == INF) return INF;

        if (i) res += dist[t];
        st[t] = true;

        for (int j = 1; j <= n; j ++ )
            // if (!st[j])
            dist[j] = min(dist[j], g[t][j]);
    }

    return res;
}
```
