# Dijkstra算法

## 最短路

单源最短路
- 所有边权都是正数
  - 朴素 Dijkstra 算法 $O(n^2)$
  - 堆优化版的 Dijkstra 算法 $O(mlogn)$
- 存在负权边
  - Bellman-Ford $O(nm)$
  - SPFA 一般 $O(m)$，最坏 $O(nm)$

多源汇最短路
- Floyd 算法 $O(n^3)$

## 朴素 Dijkstra 算法

时间复杂是 $O(n^2+m)$, n 表示点数，m 表示边数

**模板题**
- [AcWing 849. Dijkstra求最短路 I](https://www.acwing.com/problem/content/851/)
```c++
int g[N][N];  // 存储每条边
int dist[N];  // 存储1号点到每个点的最短距离
bool st[N];   // 存储每个点的最短路是否已经确定

// 求1号点到n号点的最短路，如果不存在则返回-1
int dijkstra()
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;

    for (int i = 0; i < n - 1; i ++ )
    {
        int t = -1;     // 在还未确定最短路的点中，寻找距离最小的点
        for (int j = 1; j <= n; j ++ )
            if (!st[j] && (t == -1 || dist[t] > dist[j]))
                t = j;

        // 用t更新其他点的距离
        for (int j = 1; j <= n; j ++ )
            dist[j] = min(dist[j], dist[t] + g[t][j]);

        st[t] = true;
    }

    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}
```

## 堆优化版 Dijkstra 算法

时间复杂度 $O(mlogn)$, n 表示点数，m 表示边数

**模板题**
- [AcWing 850. Dijkstra求最短路 II](https://www.acwing.com/problem/content/852/)
```c++
typedef pair<int, int> PII;

int n;      // 点的数量
int h[N], w[N], e[N], ne[N], idx;       // 邻接表存储所有边
int dist[N];        // 存储所有点到1号点的距离
bool st[N];     // 存储每个点的最短距离是否已确定

// 求1号点到n号点的最短距离，如果不存在，则返回-1
int dijkstra()
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    heap.push({0, 1});      // first存储距离，second存储节点编号

    while (heap.size())
    {
        auto t = heap.top();
        heap.pop();

        int ver = t.second, distance = t.first;

        if (st[ver]) continue;
        st[ver] = true;

        for (int i = h[ver]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (dist[j] > distance + w[i])
            {
                dist[j] = distance + w[i];
                heap.push({dist[j], j});
            }
        }
    }

    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}
```