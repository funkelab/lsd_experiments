import mala
import tensorflow as tf
import json

def create_network(input_shape, name):

    tf.reset_default_graph()

    with tf.variable_scope('setup02'):

        raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(raw, (1, 1) + input_shape)

        unet, _, _ = mala.networks.unet(
            raw_batched,
            12,
            6,
            [[2,2,2],[2,2,2],[3,3,3]],
            num_fmaps_out=14)

        lsds_batched, _ = mala.networks.conv_pass(
            unet,
            kernel_sizes=[1],
            num_fmaps=10,
            activation='sigmoid',
            name='lsds')
        # lsds = tf.squeeze(lsds_batched, axis=0)

        affs_batched, _ = mala.networks.conv_pass(
            unet,
            kernel_sizes=[1],
            num_fmaps=3,
            activation='sigmoid',
            name='affs')
        affs = tf.squeeze(affs_batched, axis=0)

        output_shape = tuple(affs.get_shape().as_list()[1:])

        lsd_shape_batched = lsds_batched.get_shape().as_list()
        lsd_output_shape = lsd_shape_batched[1:]

        lsds = tf.reshape(lsds_batched, lsd_output_shape)

        gt_lsds = tf.placeholder(tf.float32, shape=lsd_output_shape)
        loss_weights = tf.placeholder(tf.float32, shape=lsd_output_shape[1:])
        loss_weights_lsds = tf.reshape(loss_weights, (1,) + tuple(lsd_output_shape[1:]))

        gt_affs = tf.placeholder(tf.float32, shape=(3,) + output_shape)
        loss_weights_affs = tf.placeholder(tf.float32, shape=(3,) + output_shape)

        loss_lsds = tf.losses.mean_squared_error(
            gt_lsds,
            lsds,
            loss_weights_lsds)

        loss_affs = tf.losses.mean_squared_error(
            gt_affs,
            affs,
            loss_weights_affs)

        loss = loss_lsds + loss_affs

        summary = tf.summary.merge([
            tf.summary.scalar('setup02_eucl_loss', loss),
            tf.summary.scalar('setup02_eucl_loss_lsds', loss_lsds),
            tf.summary.scalar('setup02_eucl_loss_affs', loss_affs)
        ])

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
            'lsds': lsds.name,
            'gt_lsds': gt_lsds.name,
            'loss_weights_lsds': loss_weights.name,
            'affs': affs.name,
            'gt_affs': gt_affs.name,
            'loss_weights_affs': loss_weights_affs.name,
            'loss': loss.name,
            'optimizer': optimizer.name,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'summary': summary.name
        }

        config['outputs'] = {
                'affs':
                {"out_dims": 3,
                    "out_dtype": "uint8"
                    },
                'lsds':
                {"out_dims": 10,
                    "out_dtype": "uint8"
                    }
                }

        with open(name + '.json', 'w') as f:
            json.dump(config, f)


if __name__ == "__main__":

    create_network((164, 164, 164), 'train_net')
    create_network((268, 268, 268), 'config')

