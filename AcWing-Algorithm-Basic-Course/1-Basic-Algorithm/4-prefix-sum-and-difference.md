# 前缀和与差分

## 前缀和

### 一维前缀和

**模板题**
- [AcWing 795. 前缀和](https://www.acwing.com/problem/content/797/)
```c++
S[i] = a[1] + a[2] + ... a[i]
a[l] + ... + a[r] = S[r] - S[l - 1]
```

### 二维前缀和

**模板题**
- [AcWing 796. 子矩阵的和](https://www.acwing.com/problem/content/798/)
```c++
S[i, j] = 第i行j列格子左上部分所有元素的和
以(x1, y1)为左上角，(x2, y2)为右下角的子矩阵的和为：
S[x2, y2] - S[x1 - 1, y2] - S[x2, y1 - 1] + S[x1 - 1, y1 - 1]
```

## 差分

### 一维差分

**模板题**
- [AcWing 797. 差分](https://www.acwing.com/problem/content/799/)
```c++
给区间[l, r]中的每个数加上c：B[l] += c, B[r + 1] -= c
```

### 二维差分

**模板题**
- [AcWing 798. 差分矩阵](https://www.acwing.com/problem/content/800/)
```c++
给以(x1, y1)为左上角，(x2, y2)为右下角的子矩阵中的所有元素加上c：
S[x1, y1] += c, S[x2 + 1, y1] -= c, S[x1, y2 + 1] -= c, S[x2 + 1, y2 + 1] += c
```