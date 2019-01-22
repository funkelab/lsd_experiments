import mala
from mala.networks.unet import crop_zyx
import tensorflow as tf
import json

def create_network(input_shape, output_shape, name, scope='setup02'):

    tf.reset_default_graph()
    
    with tf.variable_scope(scope):

        raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(raw, (1, 1) + input_shape)

        unet, _, _ = mala.networks.unet(raw_batched, 12, 6, [[2,2,2],[2,2,2],[3,3,3]])

        embedding_batched, _ = mala.networks.conv_pass(
            unet,
            kernel_sizes=[1],
            num_fmaps=10,
            activation='sigmoid')

        embedding_batched = crop_zyx(embedding_batched, (1, 10) + output_shape)
        embedding = tf.reshape(embedding_batched, (10,) + output_shape)

        gt_embedding = tf.placeholder(tf.float32, shape=(10,)+output_shape)
        embedding_loss_weights = tf.placeholder(tf.float32, shape=(10,)+output_shape)
        loss = tf.losses.mean_squared_error(
            gt_embedding,
            embedding,
            embedding_loss_weights)

        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4,
            beta1=0.95,
            beta2=0.999,
            epsilon=1e-8)
        optimizer = opt.minimize(loss)

        print("input shape : %s"%(input_shape,))
        print("output shape: %s"%(output_shape,))

        tf.train.export_meta_graph(filename=name + '.meta')

        config = {
            'raw': raw.name,
            'embedding': embedding.name,
            'gt_embedding': gt_embedding.name,
            'embedding_loss_weights': embedding_loss_weights.name,
            'loss': loss.name,
            'optimizer': optimizer.name,
            'input_shape': input_shape,
            'output_shape': output_shape}
        with open(name + '_config.json', 'w') as f:
            json.dump(config, f)

if __name__ == "__main__":

    create_network((268, 268, 268), (164, 164, 164), 'lsd_net')
