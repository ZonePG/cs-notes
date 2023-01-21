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

**模板题**
- [AcWing 2. 01背包问题](https://www.acwing.com/problem/content/2/)
- [AcWing 3. 完全背包问题](https://www.acwing.com/problem/content/3/)
- [AcWing 4. 多重背包问题](https://www.acwing.com/problem/content/4/)
- [AcWing 5. 多重背包问题 II](https://www.acwing.com/problem/content/5/)
- [AcWing 9. 分组背包问题](https://www.acwing.com/problem/content/9/)
