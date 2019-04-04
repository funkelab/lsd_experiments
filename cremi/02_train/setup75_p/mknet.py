import mala
from mala.networks.unet import crop_zyx
import tensorflow as tf
import json

def create_auto(input_shape, output_shape, name):

    tf.reset_default_graph()

    with tf.variable_scope('setup59_p'):
        raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(raw, (1, 1) + input_shape)

        unet, _, _ = mala.networks.unet(raw_batched, 12, 5, [[1,3,3],[1,3,3],[3,3,3]])

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

def create_affs(input_shape, intermediate_shape, expected_output_shape, name):

    tf.reset_default_graph()

    with tf.variable_scope('setup75_p'):

        raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(raw, (1, 1) + input_shape)
        raw_in = tf.reshape(raw_batched, input_shape)
        raw_batched = crop_zyx(raw_batched, (1, 1) + intermediate_shape)

        raw_cropped = tf.reshape(raw_batched, intermediate_shape)

        pretrained_affs = tf.placeholder(tf.float32, shape=(3,) + intermediate_shape)
        pretrained_affs_batched = tf.reshape(pretrained_affs, (1, 3) + intermediate_shape)

        concat_input = tf.concat([raw_batched, pretrained_affs_batched], axis=1)

        unet, _, _ = mala.networks.unet(concat_input, 12, 5, [[1,3,3],[1,3,3],[3,3,3]])

        affs_batched, _ = mala.networks.conv_pass(
            unet,
            kernel_sizes=[1],
            num_fmaps=3,
            activation='sigmoid',
            name='affs2')

        affs = tf.squeeze(affs_batched, axis=0)

        output_shape = tuple(affs.get_shape().as_list()[1:])
        assert expected_output_shape == output_shape, "%s !=%s"%(expected_output_shape, output_shape)

        gt_affs = tf.placeholder(tf.float32, shape=(3,) + output_shape)
        affs_loss_weights = tf.placeholder(tf.float32, shape=(3,) + output_shape)

        loss = tf.losses.mean_squared_error(
            gt_affs,
            affs,
            affs_loss_weights)

        summary = tf.summary.scalar('setup75_eucl_loss', loss)

        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4,
            beta1=0.95,
            beta2=0.999,
            epsilon=1e-8)
        optimizer = opt.minimize(loss)

        #output_shape = output_shape[1:]
        print("input shape : %s"%(intermediate_shape,))
        print("output shape: %s"%(output_shape,))

        tf.train.export_meta_graph(filename=name + '.meta')

        config = {
            'raw': raw.name,
            'pretrained_affs': pretrained_affs.name,
            'affs': affs.name,
            'gt_affs': gt_affs.name,
            'affs_loss_weights': affs_loss_weights.name,
            'loss': loss.name,
            'optimizer': optimizer.name,
            'input_shape': intermediate_shape,
            'output_shape': output_shape,
            'summary': summary.name,
            }
        with open(name + '.json', 'w') as f:
            json.dump(config, f)

def create_config(input_shape, output_shape, num_dims, name):

    config = {
        'input_shape': input_shape,
        'output_shape': output_shape,
        'out_dims': num_dims,
        'out_dtype': 'uint8',
        'affs_setup': 'setup59_p',
        'affs_iteration': 400000
        }
    with open(name + '.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":

    z=18
    xy=162

    train_input_shape = (120, 484, 484)
    train_intermediate_shape = (84, 268, 268)
    train_output_shape = (48, 56, 56)

    create_auto(train_input_shape, train_intermediate_shape, 'train_auto_net')
    create_affs(train_input_shape, train_intermediate_shape, train_output_shape, 'train_net')

    test_input_shape = (96+z, 484+xy, 484+xy)
    test_output_shape = (78, 434, 434)

    create_affs(test_input_shape, test_input_shape, test_output_shape, 'test_net')

    create_config(test_input_shape, test_output_shape, 3, 'config')
