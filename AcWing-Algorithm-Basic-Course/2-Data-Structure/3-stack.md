# 栈

**模板题**
- [AcWing 828. 模拟栈](https://www.acwing.com/problem/content/830/)
- [AcWing 3302. 表达式求值](https://www.acwing.com/problem/content/3305/)
```c++
// tt表示栈顶
int stk[N], tt = 0;

// 向栈顶插入一个数
stk[ ++ tt] = x;

// 从栈顶弹出一个数
tt -- ;

// 栈顶的值
stk[tt];

// 判断栈是否为空
if (tt > 0)
{

}
```

## 单调栈

**模板题**
- [AcWing 830. 单调栈](https://www.acwing.com/problem/content/832/)
```c++
常见模型：找出每个数左边离它最近的比它大/小的数
int tt = 0;
for (int i = 1; i <= n; i ++ )
{
    while (tt && check(stk[tt], i)) tt -- ;
    stk[ ++ tt] = i;
}
```