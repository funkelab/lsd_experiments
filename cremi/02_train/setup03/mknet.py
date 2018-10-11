import mala
from mala.networks.unet import crop_zyx
import tensorflow as tf
import json

def create_lsd_network(input_shape, output_shape, name, scope):

    tf.reset_default_graph()

    with tf.variable_scope(scope):

        raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(raw, (1, 1) + input_shape)

        unet, _, _ = mala.networks.unet(raw_batched, 12, 5, [[1,3,3],[1,3,3],[3,3,3]])

        embedding_batched, _ = mala.networks.conv_pass(
            unet,
            kernel_sizes=[1],
            num_fmaps=10,
            activation='sigmoid')

        embedding_batched = crop_zyx(embedding_batched, (1, 10) + output_shape)
        embedding = tf.reshape(embedding_batched, (10,) + output_shape)
        embedding = embedding

        print("input shape : %s"%(input_shape,))
        print("output shape: %s"%(output_shape,))

        tf.train.export_meta_graph(filename=name + '.meta')

        config = {
            'raw': raw.name,
            'embedding': embedding.name,
            'input_shape': input_shape,
            'output_shape': output_shape}
        with open(name + '_config.json', 'w') as f:
            json.dump(config, f)

def create_affs_network(input_shape, name):

    tf.reset_default_graph()

    embedding = tf.placeholder(tf.float32, shape=(10,) + input_shape)
    embedding_batched = tf.reshape(embedding, (1, 10) + input_shape)

    unet, _, _ = mala.networks.unet(embedding_batched, 12, 5, [[1,3,3],[1,3,3],[3,3,3]])

    affs_batched, _ = mala.networks.conv_pass(
        unet,
        kernel_sizes=[1],
        num_fmaps=3,
        activation='sigmoid',
        name='affs')

    output_shape_batched = affs_batched.get_shape().as_list()
    output_shape = output_shape_batched[1:] # strip the batch dimension

    affs = tf.reshape(affs_batched, output_shape)

    gt_affs = tf.placeholder(tf.float32, shape=output_shape)
    affs_loss_weights = tf.placeholder(tf.float32, shape=output_shape)
    loss = tf.losses.mean_squared_error(
        gt_affs,
        affs,
        affs_loss_weights)

    summary = tf.summary.scalar('setup03_eucl_loss', loss)

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
        'embedding': embedding.name,
        'affs': affs.name,
        'gt_affs': gt_affs.name,
        'affs_loss_weights': affs_loss_weights.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'summary': summary.name,
        'lsd_setup': "setup02",
        'lsd_iteration': 400000
        }
    with open(name + '_config.json', 'w') as f:
        json.dump(config, f)

def create_config(input_shape, output_shape, num_dims, name):

    config = {
        'input_shape': input_shape,
        'output_shape': output_shape,
        'out_dims': num_dims
        }
    with open(name + '.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":

    create_lsd_network((120, 484, 484), (84, 268, 268), 'lsd_net', 'setup02')
    
    create_affs_network((84, 268, 268), 'train_net')
    create_affs_network((84, 268, 268), 'affs_net')
    
    create_config((120, 484, 484), (48, 56, 56), 3, 'config')

