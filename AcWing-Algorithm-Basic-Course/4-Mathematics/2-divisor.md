# 约数

## 试除法求所有约数

**模板题**
- [AcWing 869. 试除法求约数](https://www.acwing.com/problem/content/871/)
```c++
vector<int> get_divisors(int x)
{
    vector<int> res;
    for (int i = 1; i <= x / i; i ++ )
        if (x % i == 0)
        {
            res.push_back(i);
            if (i != x / i) res.push_back(x / i);
        }
    sort(res.begin(), res.end());
    return res;
}
```

## 约数个数和约数之和

**模板题**
- [AcWing 870. 约数个数](https://www.acwing.com/problem/content/872/)
- [AcWing 871. 约数之和](https://www.acwing.com/problem/content/873/)
```c++
如果 N = p1^c1 * p2^c2 * ... *pk^ck
约数个数： (c1 + 1) * (c2 + 1) * ... * (ck + 1)
约数之和： (p1^0 + p1^1 + ... + p1^c1) * ... * (pk^0 + pk^1 + ... + pk^ck)
```

## 欧几里得算法

**模板题**
- [AcWing 872. 最大公约数](https://www.acwing.com/problem/content/874/)
```c++
int gcd(int a, int b)
{
    return b ? gcd(b, a % b) : a;
}
```

