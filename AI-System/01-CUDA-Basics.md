# CUDA 编程基础

https://zhuanlan.zhihu.com/p/645330027

## CPU 与 GPU 区别

处理器最重要的两个指标：延迟和吞吐量。
- 延迟：发出指令到收到结果的时间
- 吞吐量：单位时间内处理的指令的数量

下图左边是 CPU 的结构，CPU 设计导向就是减少指令的时延，被称为延迟导向设计：
- 多级高速缓存结构，提升指令访存速度。
- 控制单元。分支预测机制和流水线前传机制。
- 运算单元 (Core) 强大，整型浮点型复杂运算速度快。

下图右边是 GPU 的结构，GPU 在设计导向是增加简单指令吞吐，被称为吞吐导向设计：
- 虽有缓存结构但是数量少。因为要减少指令访问缓存的次数。
- 控制单元非常简单。 控制单元中没有分支预测机制和数据转发机制，对于复杂的指令运算就会比较慢。
- 运算单元 (Core) 非常多，采用长延时流水线以实现高吞吐量。每一行的运算单元的控制器只有一个，意味着每一行的运算单元使用的指令是相同的，不同的是它们的数据内容。那么这种整齐划一的运算方式使得 GPU 对于那些控制简单但运算高效的指令的效率显著增加。

![](https://cdn.jsdelivr.net/gh/ZonePG/images/AISystem/202408171930464.png)

由于设计原则不同，二者擅长的场景有所不同：
- CPU 在连续计算部分，延迟优先，CPU 比 GPU 单条复杂指令延迟快10倍以上。
- GPU 在并行计算部分，吞吐优先，GPU 比 CPU 单位时间内执行指令数量10倍以上。

进一步可以具体化适合 GPU 的场景：
- 计算密集：数值计算的比例要远大于内存操作，因此内存访问的延时可以被计算掩盖。
- 数据并行：大任务可以拆解为执行相同指令的小任务，因此对复杂流程控制的需求较低。

## CUDA 结构

CUDA (Compute Unified Device Architecture) 是支持 GPU 通用计算的平台和编程模型，提供 C/C++ 语言扩展和用于编程和管理 GPU 的 API。

从硬件角度看 CUDA 内存模型：
- 基本单位是 SP（Steaming Processor），也叫 CUDA Core，是 GPU 的计算核心。每个 SP 都有自己的 register 和 local memory（属于片下内存，用于应对寄存器不足的情况）。register 和 local memory 只能被自己访问，不同的 SP 之间是彼此独立的。
- 由多个 SP 和一块 Share Memory 构成一个 SM（Streaming Multiprocessor）。share memory 可以被 SM 内的所有 SP 共享。
- 多个 SM 和一块全局内存构成 GPU。不同线程块都可以访问。

也就是说：每个 thread 都有自己的一份 register 和 local memory 的空间。同一个 block 中的每个 thread 则有共享的一份 share memory。此外，所有的 thread (包括不同 block 的 thread) 都共享一份 global memory。不同的 grid 则有各自的 global memory。

从软件的角度来讲：
- SP 对应线程 thread
- SM 对应线程块 block。块内的线程通过共享内存、原子操作和屏障同步进行协作 (shared memory, atomic operations and barrier synchronization)。不同块中的线程不能协作。
- 设备端（device）对应线程块组合体 grid

## PyTorch自定义CUDA算子

Torch 使用CUDA 算子 主要分为三个步骤：
- 先编写CUDA算子和对应的 launch 调用函数。
- 然后编写 torch cpp 函数建立 PyTorch 和 CUDA 之间的联系，用 pybind11 封装。
- 最后用 PyTorch 的 cpp 扩展库进行编译和调用。

编译及调用方法：
- JIT 编译调用，python 代码运行的时候再去编译 cpp 和 cuda 文件。`from torch.utils.cpp_extension import load`。
- SETUP 编译调用。`from torch.utils.cpp_extension import BuildExtension, CUDAExtension`。
- CMAKE 编译调用。编译生成 .so 文件，`torch.ops.load_library("build/libxxx.so")`，`torch.ops.xxx.torch_launch_xxx()` 调用。
