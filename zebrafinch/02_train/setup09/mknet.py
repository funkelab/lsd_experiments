import mala
import tensorflow as tf
import json

def create_network(input_shape, name):

    tf.reset_default_graph()

    with tf.variable_scope('setup09'):

        raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(raw, (1, 1) + input_shape)

        unet, _, _ = mala.networks.unet(
                raw_batched,
                12,
                3,
                [[1,3,3],[1,3,3],[3,3,3]])

        logits, _ = mala.networks.conv_pass(
            unet,
            kernel_sizes=[1],
            num_fmaps=1,
            activation=None,
            name='logits')

        output_shape_batched = logits.get_shape().as_list()
        output_shape = output_shape_batched[2:] # strip the batch dimension

        logits = tf.reshape(logits, output_shape)

        mask = tf.placeholder(tf.float32, shape=output_shape)
        labels = tf.placeholder(tf.float32, shape=output_shape)

        loss = tf.contrib.losses.sigmoid_cross_entropy(
            logits=logits,
            multi_class_labels=labels,
            weights=mask)

        pred_labels = tf.nn.sigmoid(logits)

        summary = tf.summary.scalar('setup09_ce_loss', loss)

        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4,
            beta1=0.95,
            beta2=0.999,
            epsilon=1e-8)
        optimizer = opt.minimize(loss)

        # output_shape = output_shape[1:]
        print("input shape : %s"%(input_shape,))
        print("Final output shape: %s"%(output_shape,))

        tf.train.export_meta_graph(filename=name + '.meta')

        config = {
            'raw': raw.name,
            'mask': mask.name,
            'labels': labels.name,
            'pred_labels': pred_labels.name,
            'logits': logits.name,
            'loss': loss.name,
            'optimizer': optimizer.name,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'summary': summary.name
            }
        config['outputs'] = {'pred_labels': {"out_dims": 1, "out_dtype": "float32"}}

        with open(name + '.json', 'w') as f:
            json.dump(config, f)

if __name__ == "__main__":

    z=0
    xy=0

    create_network((84, 268, 268), 'train_net')
    create_network((96+z, 484+xy, 484+xy), 'config')

