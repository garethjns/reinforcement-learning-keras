import tensorflow as tf


def reinforce_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return - tf.reduce_sum(y_true * tf.math.log(y_pred),
                           axis=1)
