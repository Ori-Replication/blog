
## 生物信息学基础模型

今天讲讲 生物信息学中的其他的基础模型。
ref: [Foundation models in bioinformatics | National Science Review | Oxford Academic](https://academic.oup.com/nsr/article/12/4/nwaf028/7979309)这是一篇25年的综述。基础模型也是一个很新很新的东西，也是生信近两年的研究热点。

### General FMs
基础模型（FM）代表大规模人工智能系统，这些系统首先在海量数据集上进行了广泛的预训练，从而能够应用于各种下游任务。FM 是通过在标记和未标记数据上训练神经网络来构建的，使其能够辨别基本模式并将知识泛化到新任务中。

通常来说，基础模型能够输出一个较为通用的表征（Embedding）

许多大规模基础模型分为四种 AI 模型类型：语言 FM、视觉 FM、图形 FM 和多模态 FM。
语言FM:
- word2vec, bert, GPT...
视觉FM:
- pretrained ResNet...
图 FM:
- MPNN
- GIN
多模态FM
- CLIP（视觉+语言）

### Bioinfo FMs
![[quartz/content/030 Tutorials/Language Model, Attention, Transfosrmers and Bioinfo FMs/Pasted image 20250415154025.png]]
生物信息学中的基础模型的输入可以认为有下面几种：
- DNA序列
- RNA序列
- 蛋白序列或结构
- 分子结构式
- 单细胞组学数据，如转录组等。
其也可以被归类到语言、视觉、图和多模态 FM 中。比如，序列可以用类似语言的方式标识，结构可以被用图表示，单细胞组学数据。而多模态包括对于多种组学数据的整合与应用。

![[quartz/content/030 Tutorials/Language Model, Attention, Transfosrmers and Bioinfo FMs/Pasted image 20250415154545.png]]

### 蛋白质语言模型
最经典的工作自然是得了诺贝尔奖的 AlphaFold。其属于蛋白组学相关的基础模型。其训练的时候使用了大量的蛋白质结构数据，输入蛋白序列，最终输出蛋白质的结构。但其主要功能是输出蛋白质结构， 对于蛋白质表征的能力没有那么厉害。而 ESM2 模型是专门用来做蛋白质表征的蛋白质语言模型。其通过类似于 Bert 的训练方式，就是，比如我们拿到一个蛋白质序列，给它遮住一部分，让 AI 去恢复这部分信息来进行预训练，捕捉语义信息。当然，它预训练的task应该挺多的，总之通过多种预训练，ESM2 能够学习到蛋白质的一个很厉害的表征，可能包含其理化性质、功能等等。

### 基因组(DNA)基础模型BIOINFORMATICS FM IN GENOMICS 
使用 Transformer 解码 DNA 语言已引起人们的关注，因为它可以通过通用遗传密码破译生物学功能，从而解释 DNA 是如何翻译成蛋白质的。DNABERT [ 20] 使用 Transformer 捕获基因组 DNA 序列的全局和可迁移洞察。

其可以用于多种下游任务，包括：
#### 全基因组突变效应预测(Genome-wide variant effect prediction)
在讲这个之前，我可以先介绍一下 GWAS。GWAS通过同时检测全基因组范围内的数十万至数百万个单核苷酸多态性(SNP)，寻找与特定表型（如疾病、身高、药物反应等）显著相关的遗传变异。通过这种相关性分析，我们能够通过基因测序，对人类进行遗传病风险的评估，以及分析你个人的一些特性，如你的身高可能大概在什么水平，你对一些药物是否可能发生过敏等。不知道大家有没有做过这种基因检测，可能在前几年时比较火的东西，当时我记得我和我父母就做了一个这个测序，它能预测很多指标，比如你的身高的大致范围、是否容易肥胖，甚至你的酒量，对咖啡因等化学物质的耐受程度等。这种产品也应该一直有，比如华大基因的应该是最出名的，全基因组检测。

但 GWAS 是通过统计方法来计算关联的。对于稀有突变其样本可能不足，对于罕见病的识别能力不足。同时，它分析的主要是一个相关性，而不是因果性。机器学习的一个优势是可以了解他的因果效应。比如，每个基因之间可能都有一个潜在的作用关联，它不是独立的。比如我和小明都发生了一个致病的基因突变，可能我的其他基因突变弥补了这一点，但小明就会因为这个致病的基因突变患病。因此我们需要这样一个能够做到突变效应预测的模型。相关的工作有：
- DeepSEA
- DNABERT
- DNABERT-2
- Nucleotide Transformer

#### DNA cis-regulatory region prediction  DNA 顺式调控区预测

#### DNA methylation identification DNA 甲基化鉴定


### 转录组信息学基础模型 BIOINFORMATICS FM IN TRANSCRIPTOMICS

对于RNA 进行表征学习。包括自监督学习，以及 RNA 结构预测。不知道大家了不了解 RNA，其实RNA 也可以折叠成复杂的结构。有些理论推测，在蛋白质出现之前，生物都是用 RNA 完成催化等复杂的生物学功能。这些 RNA 语言模型可以表征 RNA 序列的作用，也是非常有用。
相关工作有：

|Model|Architecture|Description|
|---|---|---|
|Language FMs|   |   |
|**RNABERT** [43]|∙ Transformer|- RNA family classification|
||∙ Pre-train + fine-tune|- Novel transcript annotation|
|||- RNA secondary structure prediction|
|**RNA-FM** [39]|∙ Transformer|- Gene expression regulation|
||∙ Pre-train + fine-tune|- SARS-CoV-2 genome evolution|
||∙ 23 million parameters|- RNA secondary structure prediction|
|**SpliceBERT** [44]|∙ Transformer|- Variant effects on splicing|
||∙ Pre-train + fine-tune|- Cross-species splice site prediction|
||∙ 19.4 million parameters|- Human genome branch point prediction|
|**RNA-MSM** [40]|∙ Transformer|- RNA solvent accessibility|
||∙ Pre-train + fine-tune|- RNA secondary structure prediction|
|**GenerRNA** [42]|∙ Transformer (decoder only)|- De novo RNA generation|
||∙ Pre-train + fine-tune|- RNA generation with specific properties|
|Vision FMs|   |   |
|**RfamGen** [41]|∙ VAE + covariance model|- Functional RNA family generation|
||∙ No pre-train|- RNA family sequence representation|
|Multimodal FMs|   |   |
|**Bert2Ome** [46]|∙ CNN + Transformer|- 2-O-methylation site prediction|
||∙ Pre-train + fine-tune||
||∙ 110 million parameters||
模型大致可以分为下面几种：
- RNA secondary structure prediction
- RNA splice site prediction
- RNA modification detection


### BIOINFORMATICS FM IN PROTEOMICS 蛋白质组学中的基础模型

| Model                    | Architecture                 | Description                            |
| ------------------------ | ---------------------------- | -------------------------------------- |
| Language FMs             |                              |                                        |
| **UniRep** [59]          | ∙ mLSTM                      | - Protein engineering                  |
|                          | ∙ Pre-train + fine-tune      | - Remote homology detection            |
|                          | ∙ 18.2 million parameters    | - Mutation effect identification       |
| **MSA Transformer** [50] | ∙ Transformer                | - Contact prediction                   |
|                          | ∙ Pre-train + fine-tune      | - Secondary structure prediction       |
|                          | ∙ 100 million parameters     |                                        |
| **ProtTrans** [51]       | ∙ Transformer                | - Protein subcellular localization     |
|                          | ∙ Pre-train + fine-tune      | - Secondary structure prediction       |
|                          | ∙ 11 billion parameters      |                                        |
| **ProteinBERT** [47]     | ∙ Transformer                | - Evolutionary: remote homology        |
|                          | ∙ Pre-train + fine-tune      | - Engineering: fluorescence, stability |
|                          | ∙ 16 million parameters      | - Secondary structure prediction       |
| **ProtGPT2** [55]        | ∙ Autoregressive transformer | - Protein sequence generation          |
|                          | ∙ Pre-train + fine-tune      |                                        |
|                          | ∙ 738 million parameters     |                                        |
| **ZymCTRL** [56]         | ∙ Transformer                | - Enzyme generation                    |
|                          | ∙ Pre-train + fine-tune      |                                        |
|                          | ∙ 700 million parameters     |                                        |
| **ESM2** [53]            | ∙ Transformer                | - Contact prediction                   |
|                          | ∙ Pre-train + fine-tune      | - Protein-protein interactions         |
|                          | ∙ 15 billion parameters      | - Evolutionary: remote homology        |
|                          |                              | - Engineering: fluorescence, stability |
|                          |                              | - Secondary structure prediction       |
| **ProGen** [57]          | ∙ Autoregressive Transformer | - Protein sequence design              |
|                          | ∙ Pre-train + fine-tune      |                                        |
|                          | ∙ 1.2 billion parameters     |                                        |
| Multimodal FMs           |                              |                                        |
| **OntoProtein** [48]     | ∙ BERT + knowledge graph     | - Contact prediction                   |
|                          | ∙ Pre-train + fine-tune      | - Protein-protein interactions         |
|                          |                              | - Evolutionary: remote homology        |
|                          |                              | - Engineering: fluorescence, stability |
|                          |                              | - Protein function prediction: GO      |
|                          |                              | - Secondary structure prediction       |
| **AlphaFold3** [24]      | ∙ Diffusion                  | - Protein-ligand interactions          |
|                          | ∙ No Pre-train               | - Protein-nucleic acid interactions    |
|                          |                              | - Antibody-antigen prediction          |
|                          |                              | - Protein structure prediction         |
|                          |                              |                                        |
比较出名的蛋白质设计工具链：

AlphaFold: 序列 -> 结构
还有一些不属于基础模型的：
RFDifussion 原子位点 -> 结构（无具体残基）
ProteinMPNN 结构 -> 序列

基础模型大致可以分为：
- Protein structure prediction
- Protein sequence generation
- Protein evolution and mutation detection
## ## BIOINFORMATICS FM IN DRUG DISCOVERY  药物研发中的生物信息学基础模型
| Model  模型                           | Architecture  建筑学                                        | Description  描述                                           |
| ----------------------------------- | -------------------------------------------------------- | --------------------------------------------------------- |
| Language FMs  语言调频                  |                                                          |                                                           |
| **SMILES-BERT** [61]  微笑-伯特 [ 61]   | ∙ BERT   ∙ 伯特                                            | - Molecular representation  <br>- 分子表征                    |
|                                     | ∙ Pre-train + fine-tune  <br>∙ 预训练 + 微调                  | - Molecular property prediction  <br>- 分子性质预测             |
| **MolGPT** [69]  分子 GPT[69]         | ∙ Transformer (decoder only)  <br>∙ Transformer（仅限解码器）   | - Molecule generation via properties  <br>- 通过属性生成分子      |
|                                     | ∙ Pre-train + fine-tune  <br>∙ 预训练 + 微调                  | - Molecule generation via scaffolds  <br>- 通过支架生成分子       |
|                                     | ∙ 6 million parameters   ∙ 600 万个参数                      |                                                           |
| **X-MOL** [62]                      | ∙ Transformer   ∙ 变压器                                    | - Molecular property prediction  <br>- 分子性质预测             |
|                                     | ∙ Pre-train + fine-tune  <br>∙ 预训练 + 微调                  | - Chemical reaction analysis  <br>- 化学反应分析                |
|                                     |                                                          | - Molecule optimization  - 分子优化                           |
| **K-BERT** [64]  K-BERT [ 64 ]      | ∙ BERT   ∙ 伯特                                            | - Atom feature prediction  <br>- 原子特征预测                   |
|                                     | ∙ Pre-train + fine-tune  <br>∙ 预训练 + 微调                  | - Molecular feature prediction  <br>- 分子特征预测              |
|                                     | ∙ 110 million parameters  <br>∙ 1.1 亿个参数                 |                                                           |
| **DrugBAN** [73]                    | ∙ FCS   ∙ FC                                             | - Drug-target pair prediction  <br>- 药物-靶标对预测             |
|                                     | ∙ No Pre-train   ∙ 无预训练                                  |                                                           |
| Vision FMs  视觉调频                    |                                                          |                                                           |
| **PMDM** [71]  聚二甲基硅氧烷 [ 71 ]       | ∙ EGNNs + SchNet   ∙ EGNN + SchNet                       | - Molecule generation of specific target  <br>- 特定目标的分子生成 |
|                                     | ∙ No pre-train   ∙ 无预训练                                  |                                                           |
| **POLYGON** [72]  多边形 [ 72]         | ∙ VAE   ∙ 可变功率放大器                                        | - Molecule generation of multi-target  <br>- 多靶点分子生成      |
|                                     | ∙ No pre-train   ∙ 无预训练                                  |                                                           |
| Graph FMs  图形 FM                    |                                                          |                                                           |
| **Mole-BERT** [65]                  | ∙ GINs   ∙ 杜松子酒                                          | - Molecular property prediction  <br>- 分子性质预测             |
|                                     | ∙ Pre-train + fine-tune  <br>∙ 预训练 + 微调                  | - Drug-target affinity prediction  <br>- 药物靶标亲和力预测        |
| **MolCLR** [67]  分子化学发光法[67]        | ∙ GNN + contrastive learning  <br>∙ GNN + 对比学习           | - Molecular representation  <br>- 分子表征                    |
|                                     | ∙ Pre-train + fine-tune  <br>∙ 预训练 + 微调                  | - Molecular property prediction  <br>- 分子性质预测             |
| **Pocket2Mol** [70]                 | ∙ MPNN                                                   | - Molecule generation via 3D pockets  <br>- 通过 3D 口袋生成分子  |
|                                     | ∙ No pre-train   ∙ 无预训练                                  |                                                           |
| **KPGT** [66]  韩国食品药品监督管理局 [ 66]    | ∙ Line Graph Transformer  <br>∙ 线图转换器                    | - Molecular representation  <br>- 分子表征                    |
|                                     | ∙ Pre-train + fine-tune  <br>∙ 预训练 + 微调                  | - Molecular property prediction  <br>- 分子性质预测             |
|                                     | ∙ 100 million parameters  <br>∙ 1 亿个参数                   |                                                           |
| **EIHGN** [74]                      | ∙ 3D GNN                                                 | - Protein-ligand binding prediction  <br>- 蛋白质-配体结合预测     |
|                                     | ∙ No pre-train   ∙ 无预训练                                  |                                                           |
| Multimodal FMs  多模态 FM              |                                                          |                                                           |
| **MoleculesSTM** [68]  分子 STM [ 68] | ∙ MolBART + GIN + SciBERT  <br>∙ MolBART + GIN + SciBERT | - Structure-text retrieval  <br>- 结构文本检索                  |
|                                     | ∙ Pre-train + fine-tune  <br>∙ 预训练 + 微调                  | - Molecule editing  - 分子编辑                                |
|                                     | ∙ 120 million parameters  <br>∙ 1.2 亿个参数                 |                                                           |

模型大致可以分为：
- Drug-like molecular property prediction
- Drug-like molecule generation
- Drug-target interaction identification

### 单细胞基础模型 BIOINFORMATICS FM IN SINGLE-CELL ANALYSIS

单细胞 RNA 测序 (scRNA-seq) 技术为众多突破铺平了道路。单细胞语言模型可用于识别细胞状态、发现新的细胞类型、推断调控网络以及整合多组学数据。基于单细胞的基础模型，我们可以自动化地为单个细胞分配生物标签，如表征它的细胞类型和状态。比如说，我们的下游任务可以是区分这个细胞是否在癌变之类的。

除了 scRNA-seq 技术，近年来单细胞多组学技术也正在迅猛发展。除了转录组数据，有些技术已经能够同时支持代谢组学等数据的加入。多组学集成能够让我们获得更全面的信息，用于进行下游任务。

近几年，随着单细胞组学技术的发展，空间组学甚至时空组学技术也正在发展。这些单细胞基础模型也可以用于揭示单细胞的空间定位。
![[quartz/content/030 Tutorials/Language Model, Attention, Transfosrmers and Bioinfo FMs/Pasted image 20250415170628.png]]

|Model|Architecture|Description|
|---|---|---|
|Language FMs|   |   |
|**scBERT** [86]|∙ BERT|- Gene-gene interaction prediction|
||∙ Pre-train + fine-tuning|- Cell type annotation|
|||- Novel cell type discovery|
|**scMVP** [77]|∙ Transformer|- Data imputation|
||∙ No pre-train|- Cell group identification|
|**scTranslator** [76]|∙ Transformer|- Gene-gene interaction prediction|
||∙ Pre-train + fine-tuning|- Gene pseudo-knockout|
|||- Cell clustering|
|**TOSICA** [85]|∙ Transformer|- Cell type annotation|
||∙ No pre-train||
|**scFoundation** [79]|∙ Transformer|- Gene expression enhancement|
||∙ Pre-train + fine-tuning|- Single-cell drug response prediction|
||∙ 100 million parameters|- Tissue drug response identification|
|**scGPT** [75]|∙ Transformer|- Cell clustering|
||∙ Pre-train + fine-tuning|- Batch correction|
|||- Gene regulatory network inference|
|**mvTCR** [88]|∙ Transformer|- Cell-level embedding|
||∙ No pre-train|- Atlas-level analysis|
|Vision FMs|   |   |
|**scButterfly** [78]|∙ VAE|- Cell type annotation|
||∙ Pre-train + fine-tuning|- Poor-quality data enhancement|
|||- Integrative multi-omics analysis|
|**MIDAS** [90]|∙ VAE + transfer learning|- Modality alignment|
||∙ No pre-train|- Data imputation|
|||- Batch correction|
|Graph FMs|   |   |
|**DeepMAPS** [87]|∙ Graph Transformer|- Cell clustering|
||∙ No pre-train|- Biological network construction|
|**SiGra** [89]|∙ Graph Transformer|- Spatial profile augmentation|
||∙ No pre-train||
|**MarsGT** [83]|∙ Graph Transformer|- Cell clustering|
||∙ No pre-train|- Gene regulatory network inference|
|**scPROTEIN** [84]|∙ GCN + contrastive learning|- Cell type annotation|
||∙ No pre-train|- Batch correction|
|||- Cell type annotation|
|||- Proteomic data exploration|
|Multimodal FMs|   |   |
|**GLUE** [22]|∙ Graph VAE|- Triple-omics data integration|
||∙ No pre-train|- Integrative regulation inference|

模型大致可以分为：
- Cell clustering
- Cell type annotation
- Multi-omics integration
## 总结
基础模型实际上能够执行的下游任务实际上远远多于我刚刚所提到的。实际上，基础模型之所以重要，就是因为其能够进行丰富的下游任务，打通很多任务的研究流程。大家也可以尽可能地发挥想象力，先理解基础模型表征了什么东西，然后再去想我们能拿这个表征来做什么，这个表征能够提供什么额外的信息等等。如果大家有机会，也可以投入基础模型的研究中去，我认为这个领域是大有可为的。