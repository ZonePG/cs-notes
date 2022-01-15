# Lab 1: IntelliJ, Java, git

我选择了 2021 春季版本的 CS61B，课程主页是 [https://sp21.datastructur.es/](https://sp21.datastructur.es/)。Calendar 有对应的学习材料以及链接。gradescope 的邀请码是 `P5WVGW`

Lab1 十分简单，就是简单熟悉一下搭建 Java 开发环境，材料 [https://sp21.datastructur.es/materials/lab/lab1/lab1](https://sp21.datastructur.es/materials/lab/lab1/lab1) 介绍十分详细。

修改`Collatz.java`文件：
```java
public static int nextNumber(int n) {
    if (n % 2 == 1) {
        return 3 * n + 1;
    } else {
        return n / 2;
    }
}
```

提交到远程仓库再在 gradescope 选择自己仓库即可，这一部分实验十分简单，主要熟悉一下环境搭建以及评分过程。