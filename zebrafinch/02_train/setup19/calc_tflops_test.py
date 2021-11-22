import csv
import sys
import tensorflow as tf
from protobuf_to_dict import protobuf_to_dict as pd

g = tf.Graph()
run_meta = tf.RunMetadata()
sess = tf.Session(graph=g)

with g.as_default():

    saver = tf.train.import_meta_graph('config.meta', clear_devices=True)
    sess.run(tf.global_variables_initializer())
    flops = tf.profiler.profile(
                g,
                run_meta=run_meta,
                #cmd='op',
                options=tf.profiler.ProfileOptionBuilder.float_operation())

print('TOTAL FLOPs = ', flops.total_float_ops)

d = pd(flops)

t = []

for i in d['children']:
    t.append(
            [
                i['name'].replace('setup19/','').replace('_','\_'),
                i['float_ops']
            ]
        )

with open('flops_breakdown.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(t)

