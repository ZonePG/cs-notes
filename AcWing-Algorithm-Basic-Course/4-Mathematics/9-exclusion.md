# 容斥原理

**模板题**
- [AcWing 890. 能被整除的数](https://www.acwing.com/problem/content/892/)
```c++
#include <iostream>
#include <algorithm>

typedef long long LL;

using namespace std;

const int N = 20;

int n, m;
int p[N];

int main() {
    cin >> n >> m;   
    
    for (int i = 0; i < m; i++) cin >> p[i];
    
    int res = 0;
    for (int i = 1; i < (1 << m); i++) {
        int t = 1, cnt = 0;
        
        for (int j = 0; j < m; j++) {
            if (i >> j & 1) {
                cnt++;
                if ((LL) t * p[j] > n) {
                    t = -1;
                    break;
                }
                t *= p[j];
            }
        }
        if (t != -1) {
            if (cnt % 2) res += n / t;   
            else res -= n / t;
        }
    }
    
    cout << res << endl;
}
```
