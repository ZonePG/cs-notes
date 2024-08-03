# 背包问题

状态表示 $f(i, j)$
- 集和
  - 所有选法
  - 条件
    - 只从前 i 个物品中选
    - 总体积 <= j
- 属性：Max，Min，数量

状态计算：集和划分
- $f(i, j)$
  - 不含 $i \le j，f(i - 1, j)$
  - 含 $i \le j，f(i - 1, j - v_i) + w_i$
  - 取 max

[AcWing 2. 01背包问题](https://www.acwing.com/problem/content/2/)
```c++
#include <iostream>

using namespace std;

const int N = 1010;

int n, m;

int v[N], w[N];

int f[N];

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        cin >> v[i] >> w[i];
    }
    
    for (int i = 1; i <= n; i++) {
        for (int j = m; j >= v[i]; j--) {
            if (v[i] <= j) f[j] = max(f[j], f[j - v[i]] + w[i]);
        }
    }
    
    cout << f[m] << endl;
    
    return 0;
}
```

[AcWing 3. 完全背包问题](https://www.acwing.com/problem/content/3/)
```c++
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int v[N], w[N];

int f[N];

int main() {
    int n, m;
    cin >> n >> m;
    
    for (int i = 1; i <= n; i++) cin >> v[i] >> w[i];
    for (int i = 1; i <= n; i++) {
        for (int j = v[i]; j <= m; j++) {
            f[j] = max(f[j], f[j - v[i]] + w[i]);
        }
    }
    
    cout << f[m] << endl;
}
```

[AcWing 4. 多重背包问题](https://www.acwing.com/problem/content/4/)

[AcWing 5. 多重背包问题 II](https://www.acwing.com/problem/content/5/)

[AcWing 9. 分组背包问题](https://www.acwing.com/problem/content/9/)
