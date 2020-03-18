############################################################################################
# https://www.javahelps.com/2019/03/serve-tensorflow-models-in-java.html

import numpy as np
from utils import utils

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.utils import build_tensor_info

from models import FaceTranslationGANInferenceModel
from face_toolbox_keras.models.verifier.face_verifier import FaceVerifier
from face_toolbox_keras.models.parser import face_parser
from face_toolbox_keras.models.detector import face_detector
from face_toolbox_keras.models.detector.iris_detector import IrisDetector

src = tf.placeholder(tf.uint8, shape=[None, None, 3], name='src') # Source image
tar = tf.placeholder(tf.uint8, shape=[None, None, 3], name='tar') # Target image
#result = tf.placeholder(tf.uint8, shape=[None, None, 3], name='result') # Result image

fn_src = "images/trump.jpg"
fns_tar = ["images/eric.jpg"]

model = FaceTranslationGANInferenceModel()

fv = FaceVerifier(classes=512)
fp = face_parser.FaceParser()
fd = face_detector.FaceAlignmentDetector()
idet = IrisDetector()

src, mask, aligned_im, (x0, y0, x1, y1), landmarks = utils.get_src_inputs(fn_src, fd, fp, idet)
tar, emb_tar = utils.get_tar_inputs(fns_tar, fd, fv)

out = model.inference(src, mask, tar, emb_tar)

result_face = np.squeeze(((out[0] + 1) * 255 / 2).astype(np.uint8))

final_face = utils.post_process_result(fn_src, fd, result_face, aligned_im, src, x0, y0, x1, y1, landmarks) 




with tf.Session() as sess:

    # Pick out the model input and output
    x_tensor = sess.graph.get_tensor_by_name('src:0')
    y_tensor = sess.graph.get_tensor_by_name('tar:0')
    #result_tensor = sess.graph.get_tensor_by_name('result:0')

    result_tensor = tf.Variable(final_face)

    src_info = build_tensor_info(x_tensor)
    tar_info = build_tensor_info(y_tensor)
    result_info = build_tensor_info(result_tensor)

    # Create a signature definition for tfserving
    signature_definition = signature_def_utils.build_signature_def(
        inputs={'src': src_info, 'tar': tar_info},
        outputs={'result': result_info},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    builder = saved_model_builder.SavedModelBuilder('face_translation_model')

    

    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_definition
        })

    # Save the model so we can serve it with a model server :)
    builder.save()