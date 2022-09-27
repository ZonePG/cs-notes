# 排序

## 快速排序——分治

L--x------R

- 确定分界点 x：q[L], q[(L+R)/2], q[R], 随机取点
- **调整区间**：|---<=x---||--->=x---|
  - 额外数组
    - a[] b[]
    - q[L~R]: q[i] <= x 的部分放到 a 数组，q[i] > x 的部分放到 b 数组
    - 将 a 数组和 b 数组复制到 q 数组中。
  - 双指针
    - i, j 分别从数组两段开始走，直到相遇，q[i] >= x, q[j] <= x 停下并交换，再继续走。
    - i 指针前面所有的数 <= x
    - j 指针后面所有的数 >= x
- 递归处理左右两段

快排是不稳定排序，若想稳定，可以双关键字排序，<q[i], i>

### AcWing 785. 快速排序 (简单)

给定你一个长度为n的整数数列。

请你使用快速排序对这个数列按照从小到大进行排序。

并将排好序的数列按顺序输出。

**输入格式**

输入共两行，第一行包含整数 n。

第二行包含 n 个整数（所有整数均在$1-10^9$范围内），表示整个数列。

**输出格式**

输出共一行，包含 n 个整数，表示排好序的数列。

**数据范围**

1 ≤ n ≤ 100000

**输入样例**：

```
5
3 1 2 4 5
```

**输出样例**：

```
1 2 3 4 5
```

**Solution**：
```c++
#include <iostream>

using namespace std;

const int N = 1e6 + 10;

int n;
int q[N];

void quick_sort(int q[], int left, int right) {
    if (left >= right) {
        // only one num or no num
        return;
    }
    int x = q[left], i = left - 1, j = right + 1;
    while (i < j) {
        do ++i; while (q[i] < x);
        do --j; while (q[j] > x);
        if (i < j) {
            swap(q[i], q[j]);
        }
    }
    quick_sort(q, left, j);
    quick_sort(q, j + 1, right);
}

int main() {
    scanf("%d", &n);
    for (int i = 0; i < n; ++i) {
        scanf("%d", &q[i]);
    }

    quick_sort(q, 0, n - 1);

    for (int i = 0; i < n; ++i) {
        printf("%d", q[i]);
    }
}
```


## 归并排序——分治

|--left--|--right--|

- 确定分界点：mid = (left + right) / 2
- 递归排序 left、right
- **归并** —— 合二为一

归并排序是稳定排序。复杂度O(nlogn)。

### 题目同785，用归并排序解决

**Solution**：
```c++
#include <iostream>

using namespace std;

const int N = 1e6 + 10;

int n;
int q[N];
int tmp[N];

void merge_sort(int q[], int left, int right) {
    if (left >= right) {
        return;
    }

    int mid = (left + right) / 2;
    merge_sort(q, left, mid);
    merge_sort(q, mid + 1, right);
    int k = 0, i = left, j = mid + 1;
    while (i <= mid && j <= right) {
        if (q[i] <= q[j]) {
            tmp[k++] = q[i++];
        } else {
            tmp[k++] = q[j++];
        }
    }
    while (i <= mid) {
        tmp[k++] = q[i++];
    }
    while (j <= right) {
        tmp[k++] = q[j++];
    }

    for (i = left, k = 0; i < right; ++i, ++k) {
        q[i] = tmp[k];
    }
}

int main() {
    scanf("%d", &n);
    for (int i = 0; i < n; ++i) {
        scanf("%d", &q[i]);
    }

    merge_sort(q, 0, n - 1);

    for (int i = 0; i < n; ++i) {
        printf("%d ", q[i]);
    }

    return 0;
}
```