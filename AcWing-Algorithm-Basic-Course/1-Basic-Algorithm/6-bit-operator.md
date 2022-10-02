# 位运算

**模板题**
- [AcWing 801. 二进制中1的个数](https://www.acwing.com/problem/content/803/)
```c++
求n的第k位数字: n >> k & 1
返回n的最后一位1：lowbit(n) = n & -n
```