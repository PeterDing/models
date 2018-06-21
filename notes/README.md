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
