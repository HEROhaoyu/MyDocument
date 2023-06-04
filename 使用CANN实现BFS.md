# 使用CANN实现BFS

之前在构思项目的时候存在一个错误逻辑：要使用NPU——》必须采用mindspore——》mindspore只支持python，而CPU代码采用C++——》需要解决c++和python之间的调度问题——》文件传输？混合编程？——》……。但是实际上第一步逻辑就错了，因为要使用NPU不一定非要采用mindspore。mindspore只是一个为了提高可编程性，降低开发成本的python框架，它是建立在CANN计算架构的基础上。由于矩阵实现的BFS算法非常简单，实际上可以绕过mindspore框架，直接用CANN的C++接口实现。

## NPU开发层次结构

![image-20230604171458374](C:\Users\HERO\AppData\Roaming\Typora\typora-user-images\image-20230604171458374.png)

### AI芯片

昇腾（HUAWEI Ascend) 310是一款高能效、灵活可编程的人工智能处理器，在典型配置下可以输出16TOPS@INT8, 8TOPS@FP16，功耗仅为8W。采用自研华为达芬奇架构，集成丰富的计算单元, 提高AI计算完备度和效率，进而扩展该芯片的适用性。全AI业务流程加速,大幅提高AI全系统的性能，有效降低部署成本。

昇腾（HUAWEI Ascend) 910是业界算力最强的AI处理器，基于自研华为达芬奇架构3D Cube技术，实现业界最佳AI性能与能效，架构灵活伸缩，支持云边端全栈全场景应用。算力方面，昇腾910完全达到设计规格，半精度（FP16）算力达到320 TFLOPS，整数精度（INT8）算力达到640 TOPS，功耗310W。

[昇腾（HUAWEI Ascend) 芯片 | 海思官网 (hisilicon.com)](https://www.hisilicon.com/cn/products/Ascend)

### 计算架构

CANN（Compute Architecture for Neural Networks）是华为针对AI场景推出的异构计算架构，对上支持多种AI框架，对下服务AI处理器与编程，发挥承上启下的关键作用，是提升昇腾AI处理器计算效率的关键平台。

[CANN | 海思官网 (hisilicon.com)](https://www.hisilicon.com/cn/products/Ascend)

### AI框架

昇思MindSpore是一个全场景深度学习框架，旨在实现易开发、高效执行、全场景覆盖三大目标。其中，易开发表现为API友好、调试难度低；高效执行包括计算效率、数据预处理效率和分布式训练效率；全场景则指框架同时支持云、边缘以及端侧场景。

[MindSpore| 海思官网 (hisilicon.com)](https://www.hisilicon.com/cn/products/Ascend)

### AI开发平台

ModelArts是面向开发者的一站式AI开发平台，为机器学习与深度学习提供海量数据预处理及半自动化标注、大规模分布式Training、自动化模型生成，及端-边-云模型按需部署能力，帮助用户快速创建和部署模型，管理全周期AI工作流。

[ModelArts (huaweicloud.com)](https://support.huaweicloud.com/modelarts/index.html)

## 在NPU上执行BFS

### 使用mindspore

mindspore提供了python接口，可以直接实现矩阵向量加，以及矩阵归约。使用比较灵活，文件读写，格式转换等操作也比较方便。代价是执行效率比较低。

```python
def BFSMatmul(x,y,eye):
    add = ops.Add()#矩阵向量加
    sum = add(x, y)                
    op = ops.ReduceMin(keep_dims=True)#按列求最小值
    output = op(sum,0)
    return output
```

### 使用CANN

#### AscendCL是什么？

**AscendCL（Ascend Computing Language）**是一套用于在昇腾平台上开发深度神经网络推理应用的C语言API库，提供运行资源管理、内存管理、模型加载与执行、算子加载与执行、媒体数据处理等API，能够实现利用昇腾硬件计算资源、在昇腾CANN平台上进行**深度学习推理计算**、**图形图像预处理**、**单算子加速计算**等能力。简单来说，就是统一的API框架，实现对所有资源的调用。

使用AscendCL在CANN上开发应用的流程

> CANN社区版文档包含应用开发（C&C++），注意较早版本只有python版本的

如果AI应用中不仅仅包括模型推理，还有数学运算（例如BLAS基础线性代数运算）、数据类型转换等功能，也想使用昇腾的算力，可以采用单算子调用的方式，直接通过AscendCL接口加载并执行单个算子，省去模型构建、训练的过程，相对轻量级，又可以使用昇腾的算力。另外，自定义的算子，也可以通过单算子调用的方式来验证算子的功能。

#### 单算子调用与模型推理的差别

在解释单算子调用与模型推理的差别前，我们先观察下面这个开发流程图，先找出基本的共同点、不同点。

- 共同点：
  - 不管是模型推理，还是单算子调用，都需要AscendCL初始化和去初始化、运行管理资源申请和释放。
  - 不管是模型推理，还是单算子调用，都涉及加载、执行的步骤，但是要注意，两者的加载、执行是调用不同的AscendCL接口。
- 不同点：
  - 模型推理涉及模型卸载的步骤，单算子调用不涉及。

**图1** 单算子调用与模型推理的流程对比

![img](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/63RC2alpha002/infacldevg/aclcppdevg/figure/zh-cn_image_0000001550544848.png)

#### 单算子调用功能开发流程

**图2** 开发流程

![zh-cn_image_0000001600969477.png](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/63RC2alpha002/infacldevg/aclcppdevg/figure/zh-cn_image_0000001600969477.png)

1. 准备环境。

   请参见[准备开发和运行环境](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha002/infacldevg/aclcppdevg/aclcppdevg_01_0004.html)。

2. 创建代码目录。

   在开发应用前，您需要先创建目录，存放代码文件、编译脚本、测试图片数据、模型文件等。

1. 编译算子时，有以下两种方式（参见[单算子调用流程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha002/infacldevg/aclcppdevg/aclcppdevg_000073.html)中的说明）：

   - 使用ATC工具编译算子生成om模型文件

     该种方式，需要先构造*.json格式单算子描述文件（描述算子的输入、输出及属性等信息），借助ATC工具，将单算子描述文件编译成om模型文件；再分别调用AscendCL接口加载om模型文件、执行算子。

     关于ATC工具的使用说明，请参见《[ATC工具使用指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha002/infacldevg/atctool/atlasatc_16_0003.html)》。

   - 也可以调用AscendCL提供的编译算子接口

     该种方式，直接调用AscendCL接口编译、执行算子。

2. 开发应用。

   依赖的头文件和库文件的说明请参见[调用接口依赖的头文件和库文件说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha002/infacldevg/aclcppdevg/aclcppdevg_000004.html#ZH-CN_TOPIC_0000001550704284__section1494913184520)。

   单算子调用的流程请参见[单算子调用流程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha002/infacldevg/aclcppdevg/aclcppdevg_000073.html)及相关的示例代码。

5. 编译运行应用，请参见[应用调试](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha002/infacldevg/aclcppdevg/aclcppdevg_000100.html)。

## 算子开发

[Ascend算子开发](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha002/operatordevelopment/ascendcopdevg/atlas_ascendc_10_0001.html)

### 样例



## 链接汇总

昇腾社区官网：

https://www.hiascend.com/

昇腾文档-昇腾社区：

https://www.hiascend.com/document?tag=community-developer

昇腾计算：

https://support.huawei.com/enterprise/zh/category/ascend-computing-pid-1557196528909?submodel=doc 

昇腾论坛：

https://bbs.huaweicloud.com/forum/forum-726-1.html 

AI开发平台ModelArts文档：

https://docs.gaoxinai.com/zh-cn/usermanual/modelarts/modelarts_01_0001_0.html

通过VSCode远程链接ModelArts开发环境

https://bbs.huaweicloud.com/blogs/280541

资源下载-昇腾社区：

https://www.hiascend.com/developer/download

异构计算架构CANN 6.0.RC1用户手册：

https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/overview/index.html