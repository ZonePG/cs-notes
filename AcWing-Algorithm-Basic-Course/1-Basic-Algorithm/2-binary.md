# 二分

## 整数二分

二分是将区间一分为二，一半满足性质，一半不满足性质

### 模版

```c++
// 区间[left, right]被划分成[1, mid]和[mid + 1, right]时使用
int bsearch_1(int left, int right) {
    while (left < right) {
        int mid = (left + right) / 2;
        if (check(mid)) right = mid; // check() 判断mid满足性质
        else left = mid + 1;
    }
    return left;
}

// 区间[left, right]被划分成[1, mid - 1]和[mid, right]时使用
int bsearch_2(int left, int right) {
    while (left < right) {
        int mid = (left + right + 1) / 2;
        if (check(mid)) left = mid;
        else right = mid - 1;
    }
    return left;
}
```

### AcWing 789. 数的范围 (简单)

给定一个按照升序排列的长度为n的整数数组，以及 q 个查询。

对于每个查询，返回一个元素k的起始位置和终止位置（位置从0开始计数）。

如果数组中不存在该元素，则返回“-1 -1”。

**输入格式**

第一行包含整数n和q，表示数组长度和询问个数。

第二行包含n个整数（均在1~10000范围内），表示完整数组。

接下来q行，每行包含一个整数k，表示一个询问元素。

**输出格式**

共q行，每行包含两个整数，表示所求元素的起始位置和终止位置。

如果数组中不存在该元素，则返回“-1 -1”。

**数据范围**

1 <= n <= 100000, 1 <= q <= 10000, 1<= k <= 10000

**输入样例**：

```r
6 3
1 2 2 3 3 4
3
4
5
```

**输出样例**：

```r
3 4
5 5
-1 -1
```

**Solution**：

```c++
#include <iostream>

using namespace std;

const int N = 1e6 + 10;

int n, m;
int q[N];
int tmp[N];

int main() {
    scanf("%d %d", &n, &m);
    for (int i = 0; i < n; ++i) {
        scanf("%d", &q[i]);
    }

    while (m--) {
        int x;
        scanf("%d", &x);

        int left = 0, right = n - 1;
        while (left < right) {
            // case 1
            int mid = (left + right) / 2;
            if (q[mid] >= x) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        if (q[left] != x) {
            cout << "-1 -1" << endl;
        } else {
            cout << left << " ";
            left = 0, right = n - 1;
            while (left < right) {
                int mid = (left + right + 1) / 2;
                if (q[mid] <= x) {
                    left = mid;
                } else {
                    right = mid - 1;
                }
            }
            cout << right << endl;
        }
    }
}

```

## 浮点数二分

### AcWing 790. 数的三次方根

```c++
#include <iostream>

using namespace std;

int main() {
    double x;
    cin >> x;

    double left = 0, right = x;
    while (left + 1e-6 < right) {
        double mid = (left + right) / 2;
        if (mid * mid * mid >= x) {
            right = mid;
        } else {
            left = mid;
        }
    }

    printf("%lf", left);
}

```