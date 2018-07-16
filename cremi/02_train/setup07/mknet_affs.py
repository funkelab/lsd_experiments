import mala
import tensorflow as tf
import json

def create_network(input_shape, name):

    tf.reset_default_graph()

    embedding = tf.placeholder(tf.float32, shape=(10,) + input_shape)
    embedding_batched = tf.reshape(embedding, (1, 10) + input_shape)

    unet = mala.networks.unet(embedding_batched, 12, 6, [[2,2,2],[2,2,2],[3,3,3]])

    affs_batched = mala.networks.conv_pass(
        unet,
        kernel_size=1,
        num_fmaps=3,
        num_repetitions=1,
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
        'output_shape': output_shape}
    with open(name + '_config.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":

    create_network((196, 196, 196), 'train_net')
    # TODO: find largest test size
    # create_network((196, 196, 196), 'test_net')
