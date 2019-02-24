import mala
from mala.networks.unet import crop_zyx
import tensorflow as tf
import json

def create_auto(input_shape, output_shape, name):

    tf.reset_default_graph()

    with tf.variable_scope('setup101_p'):

        raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(raw, (1, 1) + input_shape)

        unet, _, _ = mala.networks.unet(
                raw_batched,
                12,
                5,
                [[1,3,3],[1,3,3],[3,3,3]],
                num_fmaps_out=14)

        embedding_batched, _ = mala.networks.conv_pass(
            unet,
            kernel_sizes=[1],
            num_fmaps=10,
            activation='sigmoid',
            name='embedding')

        embedding_batched = crop_zyx(embedding_batched, (1, 10) + output_shape)
        embedding = tf.reshape(embedding_batched, (10,) + output_shape)

        print("input shape : %s"%(input_shape,))
        print("output shape: %s"%(output_shape,))

        tf.train.export_meta_graph(filename=name + '.meta')

        config = {
            'raw': raw.name,
            'embedding': embedding.name,
            'input_shape': input_shape,
            'output_shape': output_shape}
        with open(name + '.json', 'w') as f:
            json.dump(config, f)

def create_affs(input_shape, intermediate_shape, expected_output_shape, name):

    tf.reset_default_graph()

    with tf.variable_scope('setup109_p'):

        raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(raw, (1, 1) + input_shape)
        raw_in = tf.reshape(raw_batched, input_shape)
        raw_batched = crop_zyx(raw_batched, (1, 1) + intermediate_shape)
        raw_cropped = tf.reshape(raw_batched, intermediate_shape)

        pretrained_lsd = tf.placeholder(tf.float32, shape=(10,) + intermediate_shape)
        pretrained_lsd_batched = tf.reshape(pretrained_lsd, (1, 10) + intermediate_shape)

        concat_input = tf.concat([raw_batched, pretrained_lsd_batched], axis=1)

        unet, _, _ = mala.networks.unet(
                    concat_input,
                    12,
                    5,
                    [[1,3,3],[1,3,3],[3,3,3]],
                    num_fmaps_out=14)

        embedding_batched, _ = mala.networks.conv_pass(
            unet,
            kernel_sizes=[1],
            num_fmaps=10,
            activation='sigmoid',
            name='embedding')
        embedding = tf.squeeze(embedding_batched, axis=0)

        affs_batched, _ = mala.networks.conv_pass(
            unet,
            kernel_sizes=[1],
            num_fmaps=3,
            activation='sigmoid',
            name='affs')
        affs = tf.squeeze(affs_batched, axis=0)

        output_shape = tuple(affs.get_shape().as_list()[1:])
        assert expected_output_shape == output_shape, "%s !=%s"%(expected_output_shape, output_shape)

        gt_embedding = tf.placeholder(tf.float32, shape=(10,) + output_shape)
        loss_weights_embedding = tf.placeholder(tf.float32, shape=(10,) + output_shape)
        gt_affs = tf.placeholder(tf.float32, shape=(3,) + output_shape)
        loss_weights_affs = tf.placeholder(tf.float32, shape=(3,) + output_shape)

        loss_embedding = tf.losses.mean_squared_error(
            gt_embedding,
            embedding,
            loss_weights_embedding)
        loss_affs = tf.losses.mean_squared_error(
            gt_affs,
            affs,
            loss_weights_affs)
        loss = loss_embedding + loss_affs

        summary = tf.summary.merge([
            tf.summary.scalar('setup109_eucl_loss', loss),
            tf.summary.scalar('setup109_eucl_loss_lsds', loss_embedding),
            tf.summary.scalar('setup109_eucl_loss_affs', loss_affs)
            ])

        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4,
            beta1=0.95,
            beta2=0.999,
            epsilon=1e-8)
        optimizer = opt.minimize(loss)

        print("input shape : %s"%(intermediate_shape,))
        print("output shape: %s"%(output_shape,))

        tf.train.export_meta_graph(filename=name + '.meta')

        config = {
            'raw': raw.name,
            'raw_in': raw_in.name,
            'pretrained_lsd': pretrained_lsd.name,
            'embedding': embedding.name,
            'affs': affs.name,
            'gt_embedding': gt_embedding.name,
            'gt_affs': gt_affs.name,
            'loss_weights_embedding': loss_weights_embedding.name,
            'loss_weights_affs': loss_weights_affs.name,
            'loss': loss.name,
            'optimizer': optimizer.name,
            'input_shape': intermediate_shape,
            'output_shape': output_shape,
            'summary': summary.name
            }
        with open(name + '.json', 'w') as f:
            json.dump(config, f)

def create_config(input_shape, output_shape, num_dims, name):

    config = {
        'input_shape': input_shape,
        'output_shape': output_shape,
        'out_dims': num_dims,
        'out_dtype': 'uint8',
        'lsds_setup': 'setup101_p',
        'lsds_iteration': 400000
        }
    with open(name + '.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":

    z=0
    xy=0

    train_input_shape = (120, 484, 484)
    train_intermediate_shape = (84, 268, 268)
    train_output_shape = (48, 56, 56)

    create_auto(train_input_shape, train_intermediate_shape, 'train_auto_net')
    create_affs(train_input_shape, train_intermediate_shape, train_output_shape, 'train_net')

    #todo: figure out predict network shapes

    create_config(train_input_shape, train_output_shape, 3, 'config')
