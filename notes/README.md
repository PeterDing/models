# Tensorflow Models Notes

- Noise Contrastive Estimation (NCE)
    ```
    # inputs: (batch_size, embeding_dims)
    # weights: (embeding_dims, vocab_size)
    # baises: (batch_size,)
    #
    # like a dense
    logits = tf.matmul(inputs, tf.transpose(weights))
    logits = tf.nn.bias_add(logits, biases)

    labels_one_hot = tf.one_hot(labels, n_classes)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_one_hot,
        logits=logits)
    loss = tf.reduce_sum(loss, axis=1)
    ```

- skipgram negative loss
    ```
    1. 使用σ(x)=1/(1+exp(−x))函数来表达中心词wc和背景词wo同时出现在该训练数据窗口的概率：
        \mathbb{P}(D = 1 \mid w_o, w_c) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c).
    2. 给定中心词wc生成背景词wo的条件概率的对数可以近似为:
        \text{log} \mathbb{P} (w_o \mid w_c) = \text{log} \left(\mathbb{P}(D = 1 \mid w_o, w_c) \prod_{k=1, w_k \sim \mathbb{P}(w)}^K \mathbb{P}(D = 0 \mid w_k, w_c) \right).
    3. 有关给定中心词wc生成背景词wo的损失是:
        - \text{log} \mathbb{P} (w_o \mid w_c) = -\text{log} \frac{1}{1+\text{exp}(-\mathbf{u}_o^\top \mathbf{v}_c)}  - \sum_{k=1, w_k \sim \mathbb{P}(w)}^K \text{log} \frac{1}{1+\text{exp}(\mathbf{u}_{i_k}^\top \mathbf{v}_c)}.
    ```

- parallel model on GPUs
    tutorials/rnn/ptb/ptb_word_lm.py: util.auto_parallel(metagraph, m)

- tf.train.Supervisor Flow
    `tensorflow-learning/session-flow1/README.md`
