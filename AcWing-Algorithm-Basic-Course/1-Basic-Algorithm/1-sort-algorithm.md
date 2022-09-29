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

**模板题**
- [AcWing 785. 快速排序](https://www.acwing.com/problem/content/787/)
- [AcWing 786. 第k个数](https://www.acwing.com/problem/content/788/)
```c++
void quick_sort(int q[], int l, int r)
{
    if (l >= r) return;

    int i = l - 1, j = r + 1, x = q[l + r >> 1];
    while (i < j)
    {
        do i ++ ; while (q[i] < x);
        do j -- ; while (q[j] > x);
        if (i < j) swap(q[i], q[j]);
    }
    quick_sort(q, l, j), quick_sort(q, j + 1, r);
}
```

## 归并排序——分治

|--left--|--right--|

- 确定分界点：mid = (left + right) / 2
- 递归排序 left、right
- **归并** —— 合二为一

归并排序是稳定排序。复杂度O(nlogn)。

**模板题**
- [AcWing 787. 归并排序](https://www.acwing.com/problem/content/789/)
- [AcWing 788. 逆序对的数量](https://www.acwing.com/problem/content/790/)
```c++
void merge_sort(int q[], int l, int r)
{
    if (l >= r) return;

    int mid = l + r >> 1;
    merge_sort(q, l, mid);
    merge_sort(q, mid + 1, r);

    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r)
        if (q[i] <= q[j]) tmp[k ++ ] = q[i ++ ];
        else tmp[k ++ ] = q[j ++ ];

    while (i <= mid) tmp[k ++ ] = q[i ++ ];
    while (j <= r) tmp[k ++ ] = q[j ++ ];

    for (i = l, j = 0; i <= r; i ++, j ++ ) q[i] = tmp[j];
}
```