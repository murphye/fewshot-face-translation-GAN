import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    loaded = tf.saved_model.load(sess, ["serve"],"./face_translation_model")

    graph = tf.get_default_graph()
    print(graph.get_operations()[0:10])


    src = [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
    tar = [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]



    print(sess.run(fetches=['result'], feed_dict={'src:0': src, 'tar:0': tar}))