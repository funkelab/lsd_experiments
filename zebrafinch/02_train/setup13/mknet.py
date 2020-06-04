import mala
from mala.networks.unet import crop_zyx
import tensorflow as tf
import json

def create_auto(input_shape, output_shape, name):

    tf.reset_default_graph()

    with tf.variable_scope('setup01'):

        raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(raw, (1, 1) + input_shape)

        unet, _, _ = mala.networks.unet(
                raw_batched,
                12,
                5,
                [[1,3,3],[1,3,3],[3,3,3]])

        affs_batched, _ = mala.networks.conv_pass(
            unet,
            kernel_sizes=[1],
            num_fmaps=3,
            activation='sigmoid',
            name='affs')

        affs_batched = crop_zyx(affs_batched, (1, 3) + output_shape)
        affs = tf.reshape(affs_batched, (3,) + output_shape)

        print("input shape : %s"%(input_shape,))
        print("output shape: %s"%(output_shape,))

        tf.train.export_meta_graph(filename=name + '.meta')

        config = {
            'raw': raw.name,
            'affs': affs.name,
            'input_shape': input_shape,
            'output_shape': output_shape}
        with open(name + '.json', 'w') as f:
            json.dump(config, f)

def create_affs(input_shape, name):

    tf.reset_default_graph()

    with tf.variable_scope('setup13'):

        pretrained_affs = tf.placeholder(tf.float32, shape=(3,) + input_shape)
        pretrained_affs_batched = tf.reshape(pretrained_affs, (1, 3) + input_shape)

        unet, _, _ = mala.networks.unet(
                    pretrained_affs_batched,
                    12,
                    5,
                    [[1,3,3],[1,3,3],[3,3,3]])

        auto_affs_batched, _ = mala.networks.conv_pass(
            unet,
            kernel_sizes=[1],
            num_fmaps=3,
            activation='sigmoid',
            name='auto_affs')

        output_shape_batched = auto_affs_batched.get_shape().as_list()
        output_shape = output_shape_batched[1:]

        auto_affs = tf.reshape(auto_affs_batched, output_shape)

        gt_affs = tf.placeholder(tf.float32, shape=output_shape)
        loss_weights_affs = tf.placeholder(tf.float32, shape=output_shape)

        loss = tf.losses.mean_squared_error(
            gt_affs,
            auto_affs,
            loss_weights_affs)

        summary = tf.summary.scalar('setup13_eucl_loss', loss)

        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4,
            beta1=0.95,
            beta2=0.999,
            epsilon=1e-8)
        optimizer = opt.minimize(loss)

        output_shape = output_shape[1:]
        print("input shape : %s"%(input_shape,))
        print("output shape: %s"%(output_shape,))

        tf.train.export_meta_graph(filename=name + '.meta')

        config = {
            'pretrained_affs': pretrained_affs.name,
            'auto_affs': auto_affs.name,
            'gt_affs': gt_affs.name,
            'loss_weights_affs': loss_weights_affs.name,
            'loss': loss.name,
            'optimizer': optimizer.name,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'summary': summary.name
            }
        with open(name + '.json', 'w') as f:
            json.dump(config, f)

def create_config(input_shape, output_shape, name):

    config = {
        'input_shape': input_shape,
        'output_shape': output_shape,
        'affs_setup': 'setup01',
        'affs_iteration': 400000
        }

    config['outputs'] = {'affs': {"out_dims": 3, "out_dtype": "uint8"}}

    with open(name + '.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":

    z=0
    xy=0

    train_input_shape = (120, 484, 484)
    train_intermediate_shape = (84, 268, 268)
    train_output_shape = (48, 56, 56)

    create_auto(train_input_shape, train_intermediate_shape, 'train_auto_net')
    create_affs(train_intermediate_shape, 'train_net')

    test_input_shape = (96+z, 484+xy, 484+xy)
    test_output_shape = (78, 434, 434)

    create_affs(test_input_shape, 'test_net')

    create_config(test_input_shape, test_output_shape, 'config')
