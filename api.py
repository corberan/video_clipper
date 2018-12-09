import align.detect_face
import dlib
import tensorflow as tf
import numpy as np


def mtcnn_face_detector(gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction, allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    def detect_face(frame, _):
        bounding_boxes, _ = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
        return [dlib.rectangle(*[int(p) for p in np.squeeze(bounding_box[0:4])]) for bounding_box in bounding_boxes]

    return detect_face


def dlib_cnn_face_detector(model_location):
    face_detector = dlib.cnn_face_detection_model_v1(model_location)

    def detect_face(frame, upsample_num_times):
        dets = face_detector(frame, upsample_num_times)
        return [det.rect for det in dets]

    return detect_face
