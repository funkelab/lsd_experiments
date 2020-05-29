import sys
import tensorflow as tf

g = tf.Graph()
run_meta = tf.RunMetadata()
sess = tf.Session(graph=g)

with g.as_default():

    saver = tf.train.import_meta_graph('config.meta', clear_devices=True)
    sess.run(tf.global_variables_initializer())
    flops = tf.profiler.profile(
                g,
                run_meta=run_meta,
                cmd='op',
                options=tf.profiler.ProfileOptionBuilder.float_operation())

print('FLOP = ', flops.total_float_ops)
