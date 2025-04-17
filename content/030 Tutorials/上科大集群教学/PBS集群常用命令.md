---
title: PBS集群常用命令
publish: true
tags:
---


如果你是学校P40集群的用户，那么你应该可以登录到集群系统中。
[上海科技大学信息化服务网站](https://it.shanghaitech.edu.cn/gxnjs/main.htm)

在集群系统中，你能看到这样一个挺好看的前端界面
![[Pasted image 20250223142728.png]]
不过一般来说这东西没啥用，有一个比较方便的功能是可以点击运行情况概览看到每个节点的使用情况。

一般来说，我会到运行资源概览里面查看当前集群的使用情况，然后决定要申请什么卡，或者什么节点。
![[Pasted image 20250223143801.png]]

![[Pasted image 20250223144029.png]]

找到你的集群，然后就可以看到这个集群有什么卡了。

使用命令行查看当前的节点和状态：
```
pbsnodes | grep -B1 -A3 'gpu' | grep -E '^sist|gpu'
```

首先，你需要知道自己处于哪一个队列。这个在你拿到账号的时候管理员应当会和你说。用你的队列替换下面指令中的 `sist`

这里以P40集群为例。
启动交互式命令：

```
qsub -N my_task_name -l nodes=sist-gpu09 -S /bin/bash -j oe -q pub_gpu -l walltime=24:00:00 -I
```

一般我通过指定节点的方式直接申请。具体哪些节点空闲，可以到[上科大科研计算自服务平台](https://hpc.shanghaitech.edu.cn/platform/index)查看。

