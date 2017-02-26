## One-hot Representation
1. one 1 and lots of zeroes
2. 维度太大｜V｜，不易计算。
3. This word representation does not give us directly any notion of similarity.

## SVD based method ［ ］
## Probabilistic Language Model : $P(w_t|w_{1:t-1})$
1. 问题：
    Given a sequence of letters, what is the likelihood of the next letter?

2. 统计语言模型:
    - “统计语言模型把语言(词的序列)看 作一个随机事件,并赋予相应的概率来描述其属于某种语言集合 的可能性。给定一个词汇集合 V ,对于一个由 V 中的词构成的序 列S = ⟨w1, · · · , wT ⟩ ∈ Vn,统计语言模型赋予这个序列一个概率 P (S ),来衡量 S 符合自然语言的语法和语义规则的置信度。”
3. 语言模型的两个基本功能是:
    - 判断一段文本是否符合一种语言的语法和语义规则;
    - 生成符合一种语言语法或语义规则的文本。
4. 公式：
$$
  \begin{aligned}
        P(S = w_1:T) ) &= P(W_1 = w_1,W_2 = w_2,··· ,W_T = w_T)\\
        &= P (w_1 , w_2 , · · · , w_T )\\
        P(S = w_1:T ) &= P(w_1,··· ,w_T ) \\
        &= P(w_1)P(w_2|w_1)P(w_3|w_1w_2)\cdots P(w_T|w_{1:T-1})\\
        &= \prod_{t=1}^T{P(w_t|w_{1:t-1})}\\
        &(假定P(w_1) = P(w_1|w_0))
    \end{aligned}
$$

5. N-gram:
    - 我们假设一个词的概率只依赖于其前面的 n − 1 个词(n 阶马尔可夫性质),即
    $$ P(w_t|w_{1:t-1}) = P(w_{t-n+1}:w_{t-1})$$
    - 这就是N元(N-gram)语言模型。
    - 那么我们如何知道 $P(w_{t-n+1}:w_{t-1}))$ 的值呢?
这里有两种方法:
        - 一是假设语言服从多项分布,然后用最大似然估计来计 算多项分布的参数;
        - 二是用一个函数 $f(w_{1:(t−1)})$ 来直接估计下一个词是 $w_t$ 的概率。

6. 一元语言模型［ ］
7. 最大似然估计［ ］
8. 平滑技术［ ］
9. 模型评价［ ］

##Neural Network Language Model (NNLM)
1. 问题： 在统计语言模型中,一个关键的问题是估计 $P(W_t |W_{1:(t−1)} )$,即在时刻(或位置)t,给定历史信息$h_t = w_{1:(t−1)}$ 条件下,词汇表V中的每个词 $v_k(1 ≤ k \le |V|)$ 出现的概率。这个问题可以转换为一个类别数为 |V| 的多类分类问题,即:
$$
\begin{aligned}
    P_\theta(W_t = v_k|h_t = w_{1:(t-1)}) &= P_\theta(v_k|w_{1:(t-1)})\\
    &= f_k(w_{1:(t-1)},\theta)\\
\end{aligned}
\\
f_k是分类函数，估计的词汇表中第k个词出现的后验概率。θ为模型参数。
$$
2. 输入层：
    - 将语言符号序列$w_{1:(t-1)}$ 输入到神经网络模型中,首先需要将这 些符号转换为向量形式。
    - 转换方式可以通过一个词嵌入矩阵来直接映射,也叫作输入词嵌入矩阵或 查询表。词嵌入矩阵M中,第k列向量$m_k ∈ R^{d_1}$ 表示词汇表中第k个词对应的稠密向量。
    - 通过直接映射,我们得到历史信息$w_{1:(t−1)}$ 每个词对应的向量表示$v_{w_1},··· ,v_{w_t−1}$ 。
3. 隐藏层：
    - 隐藏层可以是不同类型的网络,前馈神经网络和循环神经网络,其输
入为词向量$v_{w_1},··· ,v_{w_t−1}$,输出为一个可以表示**历史信息的向量$h_t$** 。
    - 在神经网络语言模型中,常见的网络类型有以下三种:
        - 1)简单平均.($C_i$ 为每个词的权重)
        $$ h_t = \sum_{i=1}^{t-1}C_iv_{w_i}$$
        - 2)前馈神经网络[Bengio et al., 2003a]
            - 前馈神经网络要求输入的大小是固定的。
            - 拼接为一个维度为$d_1 × (n − 1)$ 的向量$x_t$。
            - 然后将 $x_t$ 输入到由多层前馈神经网络构成的隐藏层,最后一层隐藏层的输出$h_t$。
            $$h_t = x_t \oplus tanh(Wx_t + c)
        - 3)循环神经网络[Mikolov et al., 2010]
            - 和前馈神经网络不同,循环神经网络可以接受变长的输入序列.
            - 依次接受输入$v_{w_1},··· ,v_{w_t−1}$。
            - $$ h_t = tanh(Uh_{t-1} + Wv_{w_{t-1}} + c)
            - 前馈网络语言模型和循环网络语言模型的不同之处在于循环神经网络利用 隐藏状态来记录以前所有时刻的信息,而前馈神经网络只能接受前n − 1个时 刻的信息。
4. 输出层：
    - 输出层为大小为|V|,其接受的输入为历史信息的向量表示$h_t$ ,输 出为词汇表中每个词的后验概率。在神经网络语言模型中,一般使用**softmax**分类器。
    - $$y_t = softmax(\mathbf{O}h_t + b)$$
    - 其中,输出向量$y_t ∈ R|V|$ 为一个概率分布,其第k维是词汇表中第k个词出现的后验概率;
    - $O ∈ R^{|V|×d_2}$ 是最后一层隐藏层到输出层直接的权重。O也叫做输出词嵌入矩阵,矩阵中每一行也可以看作是一个词向量。
    - 在给定历史信息 $h_t$ 条件下,词汇表 V 中第 k 词 $v_k$ 出现的后验概率为:
    $$
    \begin{aligned}
    P_\theta(v_k|h_t) &= y_t^{(k)}\\
    &= softmax(s(v_k,h_t;\theta))\\
    &= \frac{exp(s(v_k,h_t;\theta))}{\Sigma_{j=1}^{|V|}exp(s(v_j,h_t;\theta))}
    \end{aligned}
    $$
    - 其中,$s(v_k,h_t;θ)=\mathbf{o}^T_k\mathbf{h_t}+b_k$ 为未归一化的得分,由神经网络计算得到;$θ$表示网络中的所有参数,包括词向量表 M 以及神经网络的权重和偏置。$\mathbf{o}_k$为输出嵌入矩阵O中的第k行向量的转置。分母为**配分函数**。
5. 训练
6. 层次化softmax: **Hierarchical Softmax**
7. 采样
8. 噪声对比估计(NCE): **Noise Contrastive Estimation**
## CBOW

## Skip-gram model

## Negative sampling

## Smoothing

## Mikolov
1. Fast: One embedding versus |C| embeddings.
2. Just read off probabilities from softmax.
3. Similar variants to CBoW possible: position specific projections.
4. Trade off between efficiency and more structured notion of context.
