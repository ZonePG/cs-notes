# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

之前有关加速注意力机制的工作是近似注意力的方法，主要关注于减少计算 FLOP（可能与实际运行时间没有关系）并且忽略了内存访问 IO 开销。实际上，在现代 GPU（V100、A100、H100、H200）上，计算速度已经超过了内存访问速度，因此当前 Transformer 的性能瓶颈卡点主要是内存访问速度。但是常见的 Python 深度学习框架（例如 PyTorch 和 Tensorflow）的接口不允许更细粒度地控制内存访问。

因此 FlashAttention 通过 CUDA 实现对内存的细粒度访问控制，主要目的是用更少的内存访问来计算注意力，避免从 HBM（实际上就是 Global Memory） 读写整个注意力矩阵。挑战包括：
- 在不访问整个输入的情况下计算 softmax reduction
- 不存储反向过程中的注意力矩阵

通过两种技术来解决这些挑战：
- **tiling**: 重构注意力计算方式，将输入分割成块，并对输入块进行多次传递，逐步执行 softmax reduction
- 存储前向过程中的 softmax 归一化因子，用于后续反向过程中在 On-chip Memory (实际上是 Share Memory) 上快速地重新计算注意力矩阵，这种方式比从 HBM 读取注意力矩阵的方法更快。
