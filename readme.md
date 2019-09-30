## 预测化合物性质的简单机器学习模型

一个用于预测周期性结构化合物带隙的机器学习模型，目的仍旧是在于了解机器学习在化学研究中的应用，同时也加深自己对机器学习的理解


## 模型介绍

模型整体较为简单，从化合物结构开始到最后计算得到的输出值的过程中，数据流动处理的过程如图

<p align="center">
    <img src="model.png" width="100%">
</p>


数据将主要经过五个处理过程，其中后四个由神经网络完成。具体如下：

1. 由化合物的晶体结构得到原子列表

依据化合物的晶体结构，可以得到晶体结构中的每一个原子和与该原子距离在一定距离内的（即截断距离）内的原子的种类，将原子种类和离中心原子的距离信息编写成一个临近列表(在这个模型中，选取的是离原子最近的19个原子，与中心原子一起组成共有20个原子的信息的列表)。因此晶体结构结构中原子的数量与临近列表的数量相同


2. 由原子的临近列表得到表示原子的向量

使用词向量方法，将列表中不同种类的元素映射为特定长度的向量，并将该向量乘以元素对应的距离的倒数，将原子类别转换为向量。每一个列表转换成了为一定长度（这里长度为20）的向量列表

3. LSTM对向量的列表处理

使用LSTM对向量列表进行处理，输出得到一个一维向量。每一个向量列表被转换成一个一维向量，因此共有与晶体结构中原子的数量相同数量的一维向量，这些一维向量组成一个新的向量列表

4. LSTM对向量的列表处理

使用LSTM对新向量列表进行处理，输出得到一个新一维向量

5. 全连接层对一维向量的处理

使用全连接层对该一维向量进行处理，输出结果值


对于一个晶体结构而言，其性质由结构中的所有原子共同决定。但是将晶体中所有的原子直接输入到单个神经网络中，一次性得到结果的方法不太容易实现。考虑到结构的性质包含了结构中的原子的贡献，那么可以先计算得到每一个原子的贡献，然后将所有原子的贡献“整合”，得到想计算的性质。其中计算单个原子的贡献和“整合”所有原子贡献的过程就由神经网络完成。

在结构中，一个原子会受到周围原子的影响，导致其贡献的改变。在上面介绍的模型中，第一步便是获得每一个原子其周围原子的信息。原子本身的性质则由一个可变的词向量表示，将表示原子性质的词向量乘以原子将距离的倒数，以简单考虑原子之间距离对其相互作用之间的影响。第三步中，使用lstm对原子与其周围的原子之间的相互作用进行计算，获得单个原子的贡献。接下来使用lstm+全连接网络“整合”了各个原子的贡献，最后得到了结构的性质结果。

在上面模型中实际只是计算了一定距离内的原子对中心原子的作用，因此本质上这个模型仍旧可以看作是建立在原子之间的局域相互作用之上的


## 数据集

### aflow数据

使用github用户rosenbrockc编写的aflow工具[1]，从aflow数据库[2]中下载得到晶体结构与对应的带隙的大小数据(带隙由DFT方法计算得到)。在下载数据中，添加了结构密度大于0.5g/cm3，计算使用PBE泛函，结构属于ICSD数据库等限定条件，以去除一些不合理的数据，最终获得60052个结构及其对应的带隙大小的数据

在模型的第一步中，设定了可处理1-85号元素的限制，最后获得28611个带隙不为零的结构数据，使用了其中的28600个数据(训练集25725个数据，测试集2875个数据)


### 其它数据集

对于周期性晶体结构而言，除开aflow数据库外，shyue ping ong课题组在文章中提供了materials projects数据库中的数据[3]，oqmd数据库[4]中也有较多的数据。

对于分子结构，qm9[5]，qm8[5]数据集中数据很多





## 结果

使用aflow数据对模型训练的得到的结果如下

<p align="center">
    <img src="MAE.png" width="100%">
</p>

其它已发表的机器学习模型的结果对比如下

|  model            |         mae(eV)|
| ------            | ------         | 
|   MEGNet[6]          |        0.32   |
|   CGCNN[7]           |         0.388 |
|   PLMF[8]            |         0.35  |
|   this model      |      0.116(train) 0.311(test)     |


由表中数据可以看到，当前模型取得了与其它已发表的模型相似的准确度



### 与其它模型的对比

MEGNet和CGCNN中，研究者均将化合物结构的原子作为节点，将原子之间的键作为边，使化合物结构抽象为一个无向图，并使用了graph network[9]网络模型对该图进行处理，获得包含原子间作用的新图。然后对新图中的节点和边进行池化，接着使用全连接层得到最后的结果。两个模型中具体的处理方法不同，但思路大致相同的。

在MEGNet和CGCNN方法抽象化合物结构为图的过程中，原子节点之间的边的确定仍旧使用原子之间的距离大小进行判断确定，这与当前模型确定临近列表中元素的方法相同。本质上，临近列表与转化的化合物结构图是等价的，但模型中的计算原子之间相互作用的方法不同，这点是模型之间的区别。graphnet网络模型中使用定义的方法对表示结构的图进行处理获得一张新的图，新图中不仅包含了原子之间的相互作用，也包含了原子之间的键信息，同时还有结构的晶格结构信息(对于周期性结构)。而在当前模型中，似乎只是考虑了原子之间的相互影响，这对应着graphnet网络模型处理后的新图中的节点部分，但也可能在当前模型中的第一步和第二步的处理过程中已经隐式考虑了键与晶格的影响，因为结构与由结构得到的临近列表是一一对应的。从前面的模型的对比，可以看到magnet和CGCNN与当前模型之间的差别与联系。


在SchNet[10]和DTNN[11]模型中，原子之间的相互作用的处理更为复杂，较当前模型中仅使用的LSTM的方法细致很多。从schnet模型的结构来看，大致的框架仍与前面介绍的模型是相似的。由于schnet模型更加复杂，因此也应当具有更强的描述能力。


## 模型的推广

在模型的构建中，输入数据仅仅是化合物的结构，而不依赖于其它的变量。该模型能够用于训练以预测带隙的大小，那么同样能够被训练用于预测化合物结构的其它性质(对于晶体结构，如弹性模量，剪切模量，热容量，热导等；对于分子，如homo，lumo，零点能，偶极矩等)，因此是一个可推广而用于预测化合物其它诸多性质的通用模型

在将模型应用于其它化合物性质时，改变模型的结构可能可以获得更好的精度，同时超参数的的调整也对模型的表现具有重要的影响。因此这里其实主要提供的是一个建立预测化合物性质的机器模型的思路或者想法，而具体的应用则需要因地制宜


## update

在前面的模型中，第三步中临近列表中元素表示的向量的排列根据与中心原子的距离由小到大排列的；而第四步中，临近列表生成的一维向量是依据临近列表对应的中心原子的坐标分数坐标x,y,z分别由小到大排列的。第四步中排列方法意味着，交换结构的x,y晶格矢量和原子的x,y分数坐标，结构本身与原结构等价，但是在第四步中获得的是不同顺序的一维向量列表。因此模型对一些结构的等价变换的操作是没有等价性的。为了了解一些等价操作对模型计算得到的结果的影响，尝试对使用的晶体结构进行了简单的变换，也就是对数据进行了一些增强，重新训练模型。

增强数据的操作是，同时交换结构的xy或yz或xz晶格矢量和原子的xy或yz或xz分数坐标，获得新的等价结构（数据较前面使用的数据更多一些，也没有限定能带带隙的大小）。如果变换得到的等价结构与原结构相同，则不加入数据集中。增强后，获得66715个训练数据（使用其中的66704个数据）和16674测试数据（使用其中的16672个），对上面得到的模型进行了训练，获得结果如下

<p align="center">
    <img src="error2.png" width="100%">
</p>

其中训练集的MAE误差为0.035，测试集MAE误差0.295，与前面模拟得到的结果相似。如果将等价结构的预测结果取平均值作为最后的结构，并于DFT结果相比较，计算误差，可以得到训练MAE误差为0.029，测试集MAE误差0.269，与非平均方法结果相近。从这结果来看，结构的等价操作不会对模型的精确度产生较大的影响。

在模型中，第三步中已经计算了周围原子对中心原子的影响，因此原子之间的作用其实已经考虑在数据中，所以原子之间已经解耦，第四步中列表的顺序对最后的结果的影响有限。这一点也许可以解释等价操作对模型的精度的没有太大影响的原因。



## 参考资料

[1] https://github.com/rosenbrockc/aflow

[2] www.aflowlib.org

[3] https://figshare.com/articles/Graphs_of_materials_project/7451351

[4] http://oqmd.org/

[5] http://quantum-machine.org/datasets/

[6] 《Graph Networks as a Universal Machine Learning
Framework for Molecules and Crystals》 https://arxiv.org/abs/1812.05055

[7] 《Crystal graph convolutional neural networks for an accurate
and interpretable prediction of material properties》  https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301

[8] 《Universal fragment descriptors for predicting
properties of inorganic crystals》  https://www.nature.com/articles/ncomms15679/

[9] 《Relational inductive biases, deep learning, and graph networks》 https://arxiv.org/abs/1806.01261?context=cs

[10] 《SchNet-A deep learning architecture for molecules and materials》 https://aip.scitation.org/doi/10.1063/1.5019779 /  https://arxiv.org/abs/1712.06113

[11] 《Quantum-chemical insights from deep tensor neural networks》  https://www.nature.com/articles/ncomms13890
