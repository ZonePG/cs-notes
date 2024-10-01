# AI Infra 面试

## C++

### 基础知识

C++ 源码到可执行文件/库文件，编译器会做哪些操作？
> - 预处理：宏替换、头文件展开、条件编译、移除注释
>   - `g++ -E source.cpp -o source.i`
> - 编译：编译器将预处理后的代码（通常是 .i 文件）转换为中间表示（例如抽象语法树），然后生成汇编代码（.s 文件）。主要操作包括语法分析（转换为抽象语法树AST）、语义分析、中间代码生成、代码优化、汇编代码生成。
>   - `g++ -S source.i -o source.s`
> - 汇编：将汇编代码（.s 文件）转换为目标代码（机器码），生成目标文件（.o 文件）。主要操作包括将汇编指令转换为机器指令、生成目标文件，包括代码和数据的二进制表示
>   - `g++ -c source.s -o source.o`
> - （对于可执行文件）链接：将目标文件（.o 文件）和库文件链接在一起，生成可执行文件或者库文件。主要操作包括符号解析（将符号引用与符号定义关联起来）、重定位（将符号引用替换为实际地址）、生成可执行文件或库文件
>   - `g++ source.o -o source`
> - （对于静态库文件）打包：使用 ar 工具将多个目标文件打包成一个静态库文件（通常是 .a 文件）。
>   - `ar rcs libmylibrary.a source1.o source2.o source3.o`
> - （对于动态库文件）链接：将目标文件链接成动态库文件（通常是 .so 文件）。动态库可以在运行时加载，因此链接时需要添加 -shared 选项。
>   - `g++ -shared -o libmylibrary.so source1.o source2.o source3.o`

C++ 静态库和动态库的区别？
> - 静态库：静态库是一种包含了多个目标文件的归档文件，通常以 .a 结尾。静态库在链接时会被整体复制到可执行文件中，因此可执行文件的体积会变大。静态库在编译时会被链接到可执行文件中，因此可执行文件在运行时不需要依赖静态库。
> - 动态库：动态库是一种包含了多个目标文件的共享库文件，通常以 .so 结尾。动态库在链接时不会被复制到可执行文件中，而是在运行时动态加载到内存中。动态库在编译时不会被链接到可执行文件中，因此可执行文件在运行时需要依赖动态库。

智能指针需要包含哪些要素？
> - unique_ptr 是一种独占式指针，保证同一时间只有一个指针指向对象；shared_ptr 是一种共享式指针，可以有多个指针指向同一个对象；weak_ptr 是一种弱引用指针，不会增加对象的引用计数，用于解决 shared_ptr 的循环引用导致的死锁问题。
> - 原始指针
> - 计数器：用于跟踪引用计数
> - 拷贝构造函数：用于增加引用计数
> - 赋值运算符重载：用于增加引用计数并减少旧指针的引用计数
> - 析构函数：计数为 0 时，用于释放资源

如果有一个 C++ 多线程程序，执行到中间卡住，如何用 gdb 找到出问题的位置？
> - attach id 关联到发生死锁的进程 id
> - info threads 查看当前进程中所有线程的信息，也可以查看到部分堆栈信息
> - thread id 进入具体线程
> - bt 查看当前线程堆栈信息

### 面向对象

对 C++ 封装和继承的理解？
> - 面向对象三个特性：封装、继承、多态
> - 封装：隐藏实现细节，使得代码模块化；把函数和数据包围起来，对不可信的进行信息隐藏。
> - 继承：继承是指一个类可以派生出一个或多个子类，子类可以继承父类的属性和方法，也可以重写父类的方法，实现代码的复用和扩展。
> - 多态：多态是指同一个函数名可以有多种不同的实现方式，可以通过覆盖（子类重新定义父类的虚函数的做法）和重载实现（允许存在多个同名函数，而这些函数的参数表不同）。

C++中public、protected、private的区别
> - 访问范围
>   - private：只能由该类的成员函数、友元的成员函数访问，不能被其他类的成员函数访问，即使是该类的对象也不能直接访问
>   - protected：可以被该类中的成员函数访问、子类中的成员函数访问、友元中的成员函数访问，但是不能被该类的对象访问
>   - public：可以被该类的成员函数、友元的成员函数、子类的成员函数访问，也可以被自己类的对象访问
> - 三种继承方式与属性变化
>   - 使用private继承,父类的所有方法在子类中变为private;
>   - 使用protected继承,父类的protected和public方法在子类中变为protected,private方法不变;
>   - 使用public继承,父类中的方法属性不发生改变;

虚函数和虚函数表的理解？
> - 虚函数：C++中的虚函数的作用主要是实现了多态的机制。用父类型别的指针指向其子类的实例，然后通过父类的指针调用实际子类的成员函数。
> - 虚函数表：虚函数表是指在每个包含虚函数的类中都存在着一个函数地址的数组。当我们用父类的指针来操作一个子类的时候，这张虚函数表指明了实际所应该调用的函数。C++的编译器保证虚函数表的指针存在于对象实例中最前面的位置，这样通过对象实例的地址得到这张虚函数表，然后就可以遍历其中函数指针，并调用相应的函数。

### 内存管理

[程序内存分布](https://www.cnblogs.com/zhouhongyuan/p/17627549.html)
> - 命令行参数和环境变量 ---- 高地址
> - 栈(stack)：向下增长
> - 堆(heap)：向上增长
> - 数据段（程序运行自动加载）
>   - .bss（未被初始化的静态变量/全局变量）
>   - .data（已被初始化的静态变量/全局变量）
> - .text（程序只读数据，程序运行自动加载）--- 低地址
>   - 常量区
>   - 代码区

内存映射 mmap
> mmap 是 用于内存映射的函数，它是一种在用户空间的程序和一个或多个文件或其他对象之间创建直接的内存访问接口的方法。通过内存映射，操作系统将一个文件或者其它对象映射到进程的地址空间中，使得文件的内容可以直接作为进程内存的一部分来读写。利用内存映射技术来打开大文件进行读写主要有以下的几点好处：
> - 内存映射允许你以字节为单位来访问文件，这使得对模型权重等二进制数据的随机访问变得更加简单和直观。
> - 对于非常大的模型文件，可能无法一次性全部加载到内存中。内存映射允许按需加载逐页部分数据，这样可以在有限的内存中处理更大的模型，大模型动辄几十个G。也就是说，内存映射并不是将打开文件中所有字节一次性读入到内存中，而是根据访问的位置进行分块读取。换句话说，使用 mmap() 后，程序可以直接在内存地址上进行操作，而不需要关心文件读写的位置。这意味着可以用简单的指针操作来替代复杂的文件偏移量管理。
> - 减少数据拷贝次数：传统的文件读取操作需要将数据从内核缓冲区复制到用户空间的缓冲区，而内存映射则避免了这种复制，通过将打开的文件直接映射到进程地址空间的办法提高了数据访问速度。

c++ 数组访问越界有可能立即 core dump 也有可能过一会 core dump，如何解释这种现象
> https://zh.wikipedia.org/wiki/%E8%A8%98%E6%86%B6%E9%AB%94%E5%8D%80%E6%AE%B5%E9%8C%AF%E8%AA%A4  
> 在使用硬件内存分段来提供虚拟内存的系统上，当硬件检测到尝试引用不存在的段、或引用段界限外的内存或引用无访问权限的内存段中的数据时，会发生储存器段错误。在仅使用内存分页的系统上，无效内存页错误通常会导致储存器段错误，而储存器段错误和内存页错误都是虚拟内存管理系统引发的错误。储存器段错误也可以独立于内存页错误发生：非法访问有效的内存页是会导致储存器段错误，而非无效内存页错误。并且段错误可能发生在内存页中间（因此没有内存页错误），例如处于同一内存页内但非法覆盖内存的缓冲区溢出。
> - 试图访问不存在的内存空间（进程内存空间以外）
> - 试图访问没有权限的内存空间（例如：访问操作系统内核的内存地址）
> - 试图写入至只读内存段（例如：代码段）

如何检测内存泄露？这些内存泄露检测工具的工作原理是什么？如何避免内存泄露？
> 检测内存泄露的方法
> - 手动检查代码：仔细检查代码中的内存分配和释放，确保每次分配内存后都有相应的释放操作。比如 malloc和free、new和delete是否配对使用了。
> - 使用调试器和工具：有一些工具可以帮助检测内存泄露。例如：
>   - **[Valgrind](https://valgrind.org/docs/manual/quick-start.html)**（仅限于Linux和macOS）：Valgrind是一个功能强大的内存管理分析工具，可以检测内存泄露、未初始化的内存访问、数组越界等问题。使用Valgrind分析程序时，只需在命令行中输入`valgrind --leak-check=yes your_program`即可。
>   - **Visual Studio中的CRT（C Runtime）调试功能**：Visual Studio提供了一些用于检测内存泄露的C Runtime库调试功能。例如，_CrtDumpMemoryLeaks函数可以在程序结束时报告内存泄露。
>   - **AddressSanitizer**：AddressSanitizer是一个用于检测内存错误的编译器插件，适用于GCC和Clang。要启用AddressSanitizer，只需在编译时添加`-fsanitize=address`选项。
>
> 内存泄露检测工具的工作原理
> - Valgrind
>   - **动态二进制插桩**：Valgrind 并不直接执行程序，而是将程序的可执行文件加载到它自己的虚拟 CPU 中运行。Valgrind 将程序的每条指令翻译为等效的代码，并在这些代码中插入额外的检查逻辑。例如，当程序执行一个内存访问时，Valgrind 会插入代码来检查该内存访问是否合法。
>   - **虚拟 CPU**：Valgrind 自带一个虚拟 CPU 模拟器，程序的所有指令在该虚拟 CPU 上执行。它并不直接在物理 CPU 上运行程序，而是通过解释器的方式来执行指令。这使得 Valgrind 可以详细追踪内存操作、线程行为等，且不用修改源代码或重新编译程序。
>   - **内存检查机制**：跟踪每个内存分配、释放操作以及内存访问，并检查常见的内存错误，如：非法读取/写入未分配的内存（如越界访问），使用未初始化的内存，重复释放内存，内存泄漏（未释放的已分配内存）。   
> - AddressSanitizer
>   - 与 Valgrind 不同，AddressSanitizer 主要通过**编译时插桩**（Compile-Time Instrumentation）来检测内存错误，而不是在程序运行时进行二进制插桩。
>   - 与 Valgrind 相比，AddressSanitizer 的运行时开销较低，通常会使程序变慢 2-3 倍，但其性能开销远小于 Valgrind（Valgrind 会使程序变慢 10-20 倍）。这是因为 ASan 主要依赖编译时插桩和影子内存进行检测，而不是在运行时对每条指令进行解释和插桩。
>
> 如何避免内存泄露
> - 使用智能指针（C++）：在C++中，可以使用智能指针（如std::unique_ptr和std::shared_ptr）来自动管理内存。这些智能指针在作用域结束时会自动释放所指向的内存，从而降低忘记释放内存或者程序异常导致内存泄露的风险。
> - 异常安全：在C++中，如果程序抛出异常，需要确保在异常处理过程中正确释放已分配的内存。使用try-catch块来捕获异常并在适当的位置释放内存。 或者使用RAII（Resource Acquisition Is Initialization）技术，将资源（如内存）的管理与对象的生命周期绑定。

## CUDA

GPU 与 CPU 区别？
> 处理器最重要的两个指标：延迟和吞吐量。
> - 延迟：发出指令到收到结果的时间
> - 吞吐量：单位时间内处理的指令的数量
> 
> 下图左边是 CPU 的结构，CPU 设计导向就是减少指令的时延，被称为延迟导向设计：
> - 多级高速缓存结构，提升指令访存速度。
> - 控制单元。分支预测机制和流水线前传机制。
> - 运算单元 (Core) 强大，整型浮点型复杂运算速度快。
> 
> 下图右边是 GPU 的结构，GPU 在设计导向是增加简单指令吞吐，被称为吞吐导向设计：
> - 虽有缓存结构但是数量少。因为要减少指令访问缓存的次数。
> - 控制单元非常简单。 控制单元中没有分支预测机制和数据转发机制，对于复杂的指令运算就会比较慢。
> - 运算单元 (Core) 非常多，采用长延时流水线以实现高吞吐量。每一行的运算单元的控制器只有一个，意味着每一行的运算单元使用的指令是相同的，不同的是它们的数据内容。那么这种整齐划一的运算方式使得 GPU 对于那些控制简单但运算高效的指令的效率显著增加。
> 
> ![](https://cdn.jsdelivr.net/gh/ZonePG/images/AISystem/202408171930464.png)
> 
> 由于设计原则不同，二者擅长的场景有所不同：
> - CPU 在连续计算部分，延迟优先，CPU 比 GPU 单条复杂指令延迟快10倍以上。
> - GPU 在并行计算部分，吞吐优先，GPU 比 CPU 单位时间内执行指令数量10倍以上。
> 
> 进一步可以具体化适合 GPU 的场景：
> - 计算密集：数值计算的比例要远大于内存操作，因此内存访问的延时可以被计算掩盖。
> - 数据并行：大任务可以拆解为执行相同指令的小任务，因此对复杂流程控制的需求较低。


介绍一下 CUDA 内存模型？
> 从硬件角度看 CUDA 内存模型：
> - 基本单位是 SP（Steaming Processor），也叫 CUDA Core，是 GPU 的计算核心。每个 SP 都有自己的 register 和 local memory（属于片下内存，用于应对寄存器不足的情况）。register 和 local memory 只能被自己访问，不同的 SP 之间是彼此独立的。
> - 由多个 SP 和一块 Share Memory 构成一个 SM（Streaming Multiprocessor）。share memory 可以被 SM 内的所有 SP 共享。
> - 多个 SM 和一块全局内存构成 GPU。不同线程块都可以访问。
> 
> 也就是说：每个 thread 都有自己的一份 register 和 local memory 的空间。同一个 block 中的每个 thread 则有共享的一份 share memory。此外，所有的 thread (包括不同 block 的 thread) 都共享一份 global memory。不同的 grid 则有各自的 global memory。
> 
> 从软件的角度来讲：
> - SP 对应线程 thread
> - SM 对应线程块 block。块内的线程通过共享内存、原子操作和屏障同步进行协作 (shared memory, atomic operations and barrier synchronization)。不同块中的线程不能协作。
> - 设备端（device）对应线程块组合体 grid


CUDA 的 block, grid, thread 的关系？
> - thread: 线程是 CUDA 程序中最基本的执行单元。每个线程都有一个唯一的线程 ID，可以通过 threadIdx.x、threadIdx.y 和 threadIdx.z 来访问。
>   - 线程是由 CUDA 核函数（kernel）中的代码并行执行的。
> - block: 块是线程的集合，每个块都有一个唯一的块 ID，可以通过 blockIdx.x、blockIdx.y 和 blockIdx.z 来访问。
>   - 线程块中的所有线程可以通过共享内存（shared memory）进行通信。
> - grid: 网格是块的集合，gridDim.x、gridDim.y 和 gridDim.z 分别表示网格包含的块的数量。
> - threadIdx 表示线程在其所在块内的索引；blockIdx 表示线程块在网格内的索引；blockDim 表示每个块中线程的数量；gridDim 表示网格中块的数量。
> - 32 个 thread 组成一个 warp，warp 是最小的调度单元，硬件会一次性把一个 warp 放在就绪的 SM 中执行。
> - 多个 thread 组成一个 block，block 更像是一个软件上的概念，一个 block 的多个 thread 是可以通过共享内存进行通信的。

如何确定 grid size 和 block size？
> - block size
>   - 首先 block size 范围是 1-1024。
>   - 考虑 occupancy 占用，要大于最大线程数量 / 最大 block 数量 (64, 96)，以及最大活跃线程数的约数。主流架构的 GPU 的 SM 最大线程数的公约数是 512，96 以上的约数还包括 128 和 256。所以可以选择 128, 256, 512。
>   - 考虑 register 数量，不能占用太多 register，所以可以选择 128，256
> - grid size
>   - element wise 程序通常 grid size 是 block size 的整数倍，这样可以保证所有的 block 都能被充分利用。

如何算 SM 利用率与 GPU 利用率？
> - SM 利用率 occupancy = 有效的线程数 / 最大线程数
> - GPU 利用率 utilization = 有效的 SM 数 / 总的 SM 数

如何理解 CUDA stream？
> - 不同 stream 之间的计算是异步的
> - cuda Stream 是一个任务的队列，你可以往里面丢 kernel，内存操作，可以通过 stream 来查看，同步这些操作。stream 内部是顺序执行，不同 stream 之间可以并行

如何理解内存墙？
> 内存墙（Memory Wall）是指处理器和内存之间速度的不匹配问题。随着处理器速度的快速增加，内存访问速度的增长却相对较慢，这导致了一个瓶颈，即内存墙。在 CUDA 编程中，内存墙通常指的是全局内存的高延迟和低带宽可能阻碍 GPU 性能的问题。为了减轻这种影响，可以通常利用共享内存来减少对全局内存的依赖。通过将频繁访问的数据缓存到共享内存中，可以减少全局内存访问的次数，从而提高性能。

如何使用 PyTorch 自定义 CUDA 算子？
> Torch 使用CUDA 算子 主要分为三个步骤：
> - 先编写CUDA算子和对应的 launch 调用函数。
> - 然后编写 torch cpp 函数建立 PyTorch 和 CUDA 之间的联系，用 pybind11 封装。
> - 最后用 PyTorch 的 cpp 扩展库进行编译和调用。
> 
> 编译及调用方法：
> - JIT 编译调用，python 代码运行的时候再去编译 cpp 和 cuda 文件。`from torch.utils.cpp_extension import load`。
> - SETUP 编译调用。`from torch.utils.cpp_extension import BuildExtension, CUDAExtension`。
> - CMAKE 编译调用。编译生成 .so 文件，`torch.ops.load_library("build/libxxx.so")`，`torch.ops.xxx.torch_launch_xxx()` 调用。

TensorCore 的输入输出数据？
> Tensor Core 在Volta、Turing、Ampere架构上输入输出数据都为和CUDA Core共享的寄存器，在Hopper架构上，为了得到更好的带宽，计算所需要的输入数据可以直接存放在共享内存上。

## 大模型计算加速

### FlashAttention

介绍一下 FlashAttention V1？
> - **Fast（with IO-Awareness），计算快**。在Flash Attention之前，也出现过一些加速Transformer计算的方法，这些方法的着眼点是“减少计算量FLOPs”，例如用一个稀疏attention做近似计算。**但是Flash attention就不一样了，它并没有减少总的计算量，因为它发现：计算慢的卡点不在运算能力，而是在读写速度上**。所以它通过降低对显存（HBM）的访问次数来加快整体运算速度，这种方法又被称为O-Awareness。Flash Attention通过分块计算（tiling）和核函数融合（kernel fusion）来降低对显存的访问。
> - **Memory Efficicent，节省显存**。在标准attention场景中，forward时我们会计算并保存N*N大小的注意力矩阵；在backward时我们又会读取它做梯度计算，这就给硬件造成了 O(N^2) 的存储压力。在Flash Attention中，则巧妙避开了这点，使得存储压力降至 O(N)。
> - **Exact Attention，精准注意力**。之前的办法会采用类似于“稀疏attention”的方法做近似。这样虽然能减少计算量，但算出来的结果并不完全等同于标准attention下的结果。但是Flash Attention却做到了完全等同于标准attention的实现方式。

FlashAttention V1 O_i 的前向过程推导？
> ![](https://cdn.jsdelivr.net/gh/ZonePG/images/AISystem/202409221832487.png)

标准 Attention 和 FlashAttention V1 前向过程计算复杂度、IO 复杂度和显存占用？
> - 计算量复杂度均为： $O\left(\frac{N^2}{B_r B_c} B_r B_c d \right) = O(N^2 d)$  
> - 原始 Attention
>   - IO 复杂度： $O\left(N d + N^2 \right)$
>   - 显存占用： $O(N^2)$
> - FlashAttention V1
>   - IO 复杂度： $O(T_c N d) = O\left(\frac{N}{B_c} N d\right) = O\left(\frac{4 N d}{M} N d\right) = O\left(\frac{N^2 d^2}{M}\right)$
>   - 显存占用： $O(N)$

介绍一下 FastAttention V2？
> V2从以下三个方面做了改进：
> - 置换内外循环位置，同时减少非矩阵的计算量。
> - 优化Attention部分thread blocks的并行化计算，新增seq_len维度的并行，使SM的利用率尽量打满。这其实也是内外循环置换这个总体思想配套的改进措施
> - 优化thread blocks内部warp级别的工作模式，尽量减少warp间的通讯和读取shared memory的次数。

### vLLM

LLM 推理有什么瓶颈？
> - 算子计算上分析：decoding 阶段，主要算子是 GEMV，它是 Memory Bound，受限于内存带宽
> - 内存容量分析：Large KV cache、Complex decoding algorithms、Complex decoding algorithms

介绍一下 vLLM？vLLM是通过什么技术，动态地为请求分配 KV cache 显存，提升显存利用率的？
> vLLM 通过一种名为 PagedAttention 的技术，动态地为请求分配 KV cache 显存，提升显存利用率。
>
> 整体上来说，PagedAttention的设计灵感来自操作系统中虚拟内存的分页管理技术。
>
> - 请求（request）可理解为操作系统中的一个进程
> - 逻辑内存（logical KV blocks）可理解为操作系统中的虚拟内存，每个block类比于虚拟内存中的一个page。每个block的大小是固定的，在vLLM中默认大小为16，即可装16个token的K/V值
> - 块表（block table）可理解为操作系统中的虚拟内存到物理内存的映射表
> - 物理内存（physical KV blocks）可理解为操作系统中的物理内存，物理块在gpu显存上，每个block类比于虚拟内存中的一个page

当采用动态分配显存的办法时，虽然明面上同一时刻能处理更多的prompt了，但因为没有为每个prompt预留充足的显存空间，如果在某一时刻整个显存被打满了，而此时所有的prompt都没做完推理，那该怎么办？
> - 当一堆请求来到vLLM服务器上时，按照 **First-Come-First-Serve（FCFS）** 原则，优先处理那些最早到来的请求。
> - 当gpu资源不足时，为了让先来的请求能尽快做完推理，**vLLM会对那些后到来的请求执行“抢占”**，即暂时终止它们的执行。
> - **一旦vLLM决定执行抢占操作，它会暂停处理新到来的请求**。在此期间，它会将被抢占的请求相关的KV block全部交换（swap）至cpu上。等交换完成后，vLLM才会继续处理新到来的请求。
> - 当vLLM认为gpu有足够资源时，它会将cpu上的KV block重新加载回gpu，恢复被抢占请求的执行（recomputation）

vLLM swapping 策略
> **问题1：该释放哪些KV cache？**  
> **问题2：要把这些KV cache释放到哪里去？**
> - **先看问题1**。由前文PagedAttention原理可知，一个请求可能对应多个block。我们既可以选择释放掉部分block，也可以选择释放掉全部block，或者更科学地，我们可以预测一下哪些block被使用的频率最低，然后释放掉这些低频block（但这种方式实现起来难度较大，性价比不是很高）。**在vLLM中，采取的是all-or-nothing策略，即释放被抢占请求的所有block。**
> - **再来看问题2。对于这些被选中要释放的KV block**，如果将它们直接丢掉，那未免过于浪费。**vLLM采用的做法是将其从gpu上交换（Swap）到cpu上**。这样等到gpu显存充份时，再把这些block从cpu上重载回来。

vLLM recomputation 策略
> 知道了Swapping机制，重计算的过程也很好理解了：对于有些任务（比如parallel sampling中并行采样数n=1的任务），当它们因为资源不足而被抢占时，可以不做swap，而是直接释放它们的物理块，把它们重新放入等待处理的队列中，等后续资源充足时再重新从prefill阶段开始做推理
