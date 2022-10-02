# 双指针算法

**模板题**
- [AcWing 799. 最长连续不重复子序列](https://www.acwing.com/problem/content/801/)
- [AcWing 800. 数组元素的目标和](https://www.acwing.com/problem/content/802/)
- [AcWing 2816. 判断子序列](https://www.acwing.com/problem/content/2818/)
```c++
for (int i = 0, j = 0; i < n; i ++ )
{
    while (j < i && check(i, j)) j ++ ;

    // 具体问题的逻辑
}
常见问题分类：
    (1) 对于一个序列，用两个指针维护一段区间
    (2) 对于两个序列，维护某种次序，比如归并排序中合并两个有序序列的操作
```