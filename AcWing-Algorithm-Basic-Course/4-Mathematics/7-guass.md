# 高斯消元

**模板题**
- [AcWing 883. 高斯消元解线性方程组](https://www.acwing.com/problem/content/885/)
```c++
// a[N][N]是增广矩阵
int gauss()
{
    int c, r;
    for (c = 0, r = 0; c < n; c ++ )
    {
        int t = r;
        for (int i = r; i < n; i ++ )   // 找到绝对值最大的行
            if (fabs(a[i][c]) > fabs(a[t][c]))
                t = i;

        if (fabs(a[t][c]) < eps) continue;

        for (int i = c; i <= n; i ++ ) swap(a[t][i], a[r][i]);      // 将绝对值最大的行换到最顶端
        for (int i = n; i >= c; i -- ) a[r][i] /= a[r][c];      // 将当前行的首位变成1
        for (int i = r + 1; i < n; i ++ )       // 用当前行将下面所有的列消成0
            if (fabs(a[i][c]) > eps)
                for (int j = n; j >= c; j -- )
                    a[i][j] -= a[r][j] * a[i][c];

        r ++ ;
    }

    if (r < n)
    {
        for (int i = r; i < n; i ++ )
            if (fabs(a[i][n]) > eps)
                return 2; // 无解
        return 1; // 有无穷多组解
    }

    for (int i = n - 1; i >= 0; i -- )
        for (int j = i + 1; j < n; j ++ )
            a[i][n] -= a[i][j] * a[j][n];

    return 0; // 有唯一解
}
```
- [AcWing 884. 高斯消元解异或线性方程组](https://www.acwing.com/problem/content/886/)
```c++
#include <iostream>

using namespace std;

const int N = 110;

int a[N][N];

int n;

int gauss() {
    int r, c;
    for (r = 0, c = 0; c < n; c++) {
        int t = r;
        for (int i = r; i < n; i++) {
            if (a[i][c]) {
                t = i;
                break;
            }
        }
        
        if (!a[t][c]) continue;
        
        for (int i = c; i <= n; i++) swap(a[t][i], a[r][i]);
        for (int i = r + 1; i < n; i++) {
            if (a[i][c]) {
                for (int j = n; j >= c; j--) {
                    a[i][j] ^= a[r][j];
                }
            }
        }
        r++;
    }
    
    if (r < n) {
        for (int i = r; i < n; i++) {
            if (a[i][n]) {
                return 2;
            }
        }
        return 1;
    }
    
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            a[i][n] ^= a[i][j] & a[j][n];
        }
    }
    
    return 0;
}

int main() {
    cin >> n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n + 1; j++) {
            cin >> a[i][j];
        }
    }
    
    int res = gauss();
    
    if (res == 0) {
        for (int i = 0; i < n; i++) cout << a[i][n] << endl;
    } else if (res == 1) {
        cout << "Multiple sets of solutions" << endl;
    } else {
        cout << "No solution" << endl;
    }
}
```