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

from keras import backend as K

#src_input = tf.placeholder(tf.uint8, shape=[None, None, 3], name='src') # Source image
#tar_input = tf.placeholder(tf.uint8, shape=[None, None, 3], name='tar') # Target image

#tf.config.experimental_run_functions_eagerly(True)

#tf.compat.v1.disable_v2_behavior()

@tf.function(input_signature=[tf.TensorSpec([], tf.uint8), tf.TensorSpec([], tf.uint8)])
def face_translation(src_in, tar_in):

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

    result_face_final = utils.post_process_result(fn_src, fd, result_face, aligned_im, src, x0, y0, x1, y1, landmarks)
    print(">>> Shape: ", result_face_final.shape, result_face_final[0][0].dtype)

    return result_face_final

with tf.Graph().as_default():
    g = tf.Graph();
  
    with tf.Session(graph=g) as sess:
        
            to_export = tf.Module()
            to_export.face_translation = face_translation

            tf.saved_model.save(to_export, "face_translation_model")

    #final_result = face_translation(src_input, tar_input)
    #result = tf.Variable(final_result)

    #sess.run(tf.global_variables_initializer()) 

    # Pick out the model input and output
    #src_tensor = sess.graph.get_tensor_by_name('src:0')
    #tar_tensor = sess.graph.get_tensor_by_name('tar:0')
    #result_tensor = sess.graph.get_tensor_by_name('result:0')

    #src_info = build_tensor_info(src_tensor)
    #tar_info = build_tensor_info(tar_tensor)
    #result_info = build_tensor_info(result)

    # Create a signature definition for tfserving
    #signature_definition = signature_def_utils.build_signature_def(
    #    inputs={'src': src_info, 'tar': tar_info},
    #    outputs={'result': result_info},
    #    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    #builder = saved_model_builder.SavedModelBuilder('face_translation_model')

    #builder.add_meta_graph_and_variables(
    #    sess, [tag_constants.SERVING],
    #    signature_def_map={
    #        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
    #            signature_definition
    #    })

    # Save the model so we can serve it with a model server
    #builder.save()