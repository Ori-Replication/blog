---
title: 上科大信院AI集群normal队列显卡介绍
publish: true
tags:
---


以下是这些显卡与深度学习相关的关键参数对比：

1. **NVIDIA GeForce RTX 2080 Ti**
   - 架构: Turing
   - CUDA核心: 4352
   - 显存: 11GB GDDR6
   - 显存带宽: 616 GB/s
   - Tensor核心: 544 (支持FP16/INT8/INT4)
   - 深度学习性能: 14.2 TFLOPS (FP16)
   - 特点: 支持实时光线追踪，适合中等规模深度学习

2. **NVIDIA GeForce GTX 1080**
   - 架构: Pascal
   - CUDA核心: 2560
   - 显存: 8GB GDDR5X
   - 显存带宽: 320 GB/s
   - 深度学习性能: 8.9 TFLOPS (FP16)
   - 特点: 经典Pascal架构，适合入门级深度学习

3. **NVIDIA TITAN X (Pascal)**
   - 架构: Pascal
   - CUDA核心: 3584
   - 显存: 12GB GDDR5X
   - 显存带宽: 480 GB/s
   - 深度学习性能: 11 TFLOPS (FP16)
   - 特点: 大显存适合大规模模型训练

4. **Tesla V100-PCIE-16GB**
   - 架构: Volta
   - CUDA核心: 5120
   - 显存: 16GB HBM2
   - 显存带宽: 900 GB/s
   - Tensor核心: 640 (支持FP16/FP32/INT8/INT4)
   - 深度学习性能: 125 TFLOPS (Tensor Core FP16)
   - 特点: 专业级深度学习卡，支持混合精度计算

5. **Tesla M40 24GB**
   - 架构: Maxwell
   - CUDA核心: 3072
   - 显存: 24GB GDDR5
   - 显存带宽: 288 GB/s
   - 深度学习性能: 7 TFLOPS (FP32)
   - 特点: 大显存适合大规模推理任务

6. **NVIDIA TITAN V**
   - 架构: Volta
   - CUDA核心: 5120
   - 显存: 12GB HBM2
   - 显存带宽: 653 GB/s
   - Tensor核心: 640 (支持FP16/FP32/INT8/INT4)
   - 深度学习性能: 110 TFLOPS (Tensor Core FP16)
   - 特点: 消费级Volta架构，性能接近Tesla V100

**总结对比：**
- **性能最强**: Tesla V100-PCIE-16GB
- **性价比最高**: RTX 2080 Ti
- **大显存选择**: Tesla M40 24GB
- **入门级选择**: GTX 1080
- **专业级选择**: Tesla V100 或 TITAN V

这些显卡都支持CUDA和cuDNN，可以用于深度学习训练和推理，但专业级显卡（Tesla系列）在稳定性和ECC内存支持上更有优势。
