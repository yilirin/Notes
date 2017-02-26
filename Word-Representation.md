## One-hot Representation
1. one 1 and lots of zeroes
2. 维度太大｜V｜，不易计算。
3. This word representation does not give us directly any notion of similarity.

## SVD based method ［ ］
## N-gram model : $P(w_t|w_{1:t-1})$
Given a sequence of letters (for example, the sequence "for ex"), what is the likelihood of the next letter?



1. 统计语言模型:
    - “统计语言模型把语言(词的序列)看 作一个随机事件,并赋予相应的概率来描述其属于某种语言集合 的可能性。给定一个词汇集合 V ,对于一个由 V 中的词构成的序 列S = ⟨w1, · · · , wT ⟩ ∈ Vn,统计语言模型赋予这个序列一个概率 P (S ),来衡量 S 符合自然语言的语法和语义规则的置信度。”
2. 语言模型的两个基本功能是:
    - 判断一段文本是否符合一种语言的语法和语义规则;
    - 生成符合一种语言语法或语义规则的文本。
3. 公式：
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

4. N-gram:
    - 我们假设一个词的概率只依赖于其前面的 n − 1 个词(n 阶马尔可夫性质),即
    $$ P(w_t|w_{1:t-1}) = P(w_{t-n+1}:w_{t-1})$$
    - 这就是N元(N-gram)语言模型。
    - 那么我们如何知道 $P(w_{t-n+1}:w_{t-1}))$ 的值呢?
这里有两种方法:
        - 一是假设语言服从多项分布,然后用最大似然估计来计 算多项分布的参数;
        - 二是用一个函数 $f(w_{1:(t−1)})$ 来直接估计下一个词是 $w_t$ 的概率。

5. 一元语言模型［ ］
6. 最大似然估计［ ］
7. 平滑技术［ ］
8. 模型评价［ ］

## CBOW

## Skip-gram model

## Negative sampling

## Smoothing

## Mikolov
1. Fast: One embedding versus |C| embeddings.
2. Just read off probabilities from softmax.
3. Similar variants to CBoW possible: position specific projections.
4. Trade off between efficiency and more structured notion of context.
