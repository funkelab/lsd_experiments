import mala
import tensorflow as tf
import json

def create_network(input_shape, name):

    tf.reset_default_graph()

    with tf.variable_scope('setup07'):

        raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(raw, (1, 1) + input_shape)

        unet, _, _ = mala.networks.unet(
            raw_batched,
            12,
            6,
            [[1,3,3],[1,3,3],[3,3,3]])

        affs_batched, _ = mala.networks.conv_pass(
            unet,
            kernel_sizes=[1],
            num_fmaps=3,
            activation='sigmoid',
            name='affs')
        affs = tf.squeeze(affs_batched, axis=0)

        output_shape = tuple(affs.get_shape().as_list()[1:])

        gt_affs = tf.placeholder(tf.float32, shape=(3,) + output_shape)
        loss_weights_affs = tf.placeholder(tf.float32, shape=(3,) + output_shape)

        loss = tf.losses.mean_squared_error(
            gt_affs,
            affs,
            loss_weights_affs)

        summary = tf.summary.scalar('setup07_eucl_loss', loss)

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
            'affs': affs.name,
            'gt_affs': gt_affs.name,
            'loss_weights_affs': loss_weights_affs.name,
            'loss': loss.name,
            'optimizer': optimizer.name,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'summary': summary.name
        }

        config['outputs'] = {'affs': {'out_dims': 3, 'out_dtype': 'uint8'}}

        with open(name + '.json', 'w') as f:
            json.dump(config, f)


if __name__ == "__main__":

    z=0
    xy=0

    create_network((84, 268, 268), 'train_net')
    create_network((96+z, 484+xy, 484+xy), 'config')
