# 时空复杂度分析

一般ACM或者笔试题的时间限制是1秒或2秒。

在这种情况下，C++ 代码中的操作次数控制在 $10^7 \sim 10^8$ 为最佳。

下面给出在不同数据范围下，代码的时间复杂度和算法该如何选择：
- $n \le 30$，指数级别，dfs+剪枝，状态压缩dp
- $n \le 100 \Rightarrow O(n^3)$，floyd，dp，高斯消元
- $n \le 1000 \Rightarrow O(n^2), O(n^2logn)$，dp，二分，朴素版Dijkstra、朴素版Prim、Bellman-Ford
- $n \le 10000 \Rightarrow O(n \cdot \sqrt{n})$，块状链表、分块、莫队
- $n \le 100000 \Rightarrow O(nlogn)$，各种sort，线段树、树状数组、set/map、heap、拓扑排序、dijkstra+heap、prim+heap、Kruskal、spfa、求凸包、求半平面交、二分、CDQ分治、整体二分、后缀数组、树链剖分、动态树
- $n \le 1000000 \Rightarrow O(n)$，以及常数较小的 $O(nlogn)$ 算法，单调队列、 hash、双指针扫描、并查集，kmp、AC自动机，常数比较小的 $O(nlogn)$ 的做法：sort、树状数组、heap、dijkstra、spfa
- $n \le 10000000 \Rightarrow O(n)$，双指针扫描、kmp、AC自动机、线性筛素数
- $n \le 10^9 \Rightarrow O(\sqrt{n})$，判断质数
- $n \le 10^{18} \Rightarrow O(logn)$，最大公约数，快速幂，数位 DP
- $n \le 10^{1000} \Rightarrow O((logn)^2)$，高精度加减乘除
- $n \le 10^{100000} \Rightarrow O(logk \cdot loglogk)$，k 表示位数，高精度加减、FFT/NTT