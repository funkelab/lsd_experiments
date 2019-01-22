import mala
import tensorflow as tf
import json

def create_network(input_shape, name, scope, make_config=False):

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

        output_shape_batched = embedding_batched.get_shape().as_list()
        output_shape = output_shape_batched[1:] # strip the batch dimension

        embedding = tf.reshape(embedding_batched, output_shape)

        gt_embedding = tf.placeholder(tf.float32, shape=output_shape)
        embedding_loss_weights = tf.placeholder(tf.float32, shape=output_shape)
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

        output_shape = output_shape[1:]
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
    
        if make_config:
            config = {
                'raw': raw.name,
                'embedding': embedding.name,
                'gt_embedding': gt_embedding.name,
                'embedding_loss_weights': embedding_loss_weights.name,
                'loss': loss.name,
                'optimizer': optimizer.name,
                'input_shape': input_shape,
                'output_shape': output_shape,
                'out_dims': 10,
                'out_dtype': 'float32'
                }
            with open('config.json', 'w') as f:
                json.dump(config, f)

if __name__ == "__main__":

    create_network((196, 196, 196), 'train_net', 'setup02')
    create_network((248, 248, 248), 'test_net', 'setup02', make_config=True)
