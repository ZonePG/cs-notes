# 扩展欧几里得算法

**模板题**
- [AcWing 877. 扩展欧几里得算法](https://www.acwing.com/problem/content/879/)
- [AcWing 878. 线性同余方程](https://www.acwing.com/problem/content/880/)
```c++
// 求x, y，使得ax + by = gcd(a, b)
int exgcd(int a, int b, int &x, int &y)
{
    if (!b)
    {
        x = 1; y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= (a/b) * x;
    return d;
}
```