# 质数

## 试除法判定质数

**模板题**
- [AcWing 866. 试除法判定质数](https://www.acwing.com/problem/content/868/)
```c++
bool is_prime(int x)
{
    if (x < 2) return false;
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
            return false;
    return true;
}
```

## 试除法分解质因数

**模板题**
- [AcWing 867. 分解质因数](https://www.acwing.com/problem/content/869/)
```c++
void divide(int x)
{
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
        {
            int s = 0;
            while (x % i == 0) x /= i, s ++ ;
            cout << i << ' ' << s << endl;
        }
    if (x > 1) cout << x << ' ' << 1 << endl;
    cout << endl;
}
```

## 朴素筛法求素数

**模板题**
- [AcWing 868. 筛质数](https://www.acwing.com/problem/content/870/)
```c++
int primes[N], cnt;     // primes[]存储所有素数
bool st[N];         // st[x]存储x是否被筛掉

void get_primes(int n)
{
    for (int i = 2; i <= n; i ++ )
    {
        if (st[i]) continue;
        // 只删质数
        primes[cnt ++ ] = i;
        for (int j = i + i; j <= n; j += i)
            st[j] = true;
    }
}
```

## 线性筛法求素数 

x 只会被最小质因子筛掉
- i % prime[j] == 0
  - pj 一定是 i 的最小质因子，pj 一定是 pj * i 的最小质因子
- i % prime[j] != 0
  - pj 一定是小于 i 的所有质因子，pj 也一定是 pj * i 的最小质因子

**模板题**
- [AcWing 868. 筛质数](https://www.acwing.com/problem/content/870/)
```c++
int primes[N], cnt;     // primes[]存储所有素数
bool st[N];         // st[x]存储x是否被筛掉

void get_primes(int n)
{
    for (int i = 2; i <= n; i ++ )
    {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= n / i; j ++ )
        {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
}
```