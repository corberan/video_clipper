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

    def detect_faces(frames, upsample_num_times: int):
        for frame in frames:
            if upsample_num_times > 0:
                frame = dlib.resize_image(frame, scale=upsample_num_times)
            bounding_boxes, _ = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            yield (dlib.rectangle(*[int(p) for p in np.squeeze(bounding_box[0:4])]) for bounding_box in
                   bounding_boxes), frame

    return detect_faces


def dlib_cnn_face_detector(model_location):
    face_detector = dlib.cnn_face_detection_model_v1(model_location)

    def detect_faces(frames, upsample_num_times: int):
        frame_list = [frame for frame in frames]
        frame_dets_list = face_detector(frame_list, upsample_num_times)
        return zip([frame_dets for frame_dets in frame_dets_list], frame_list)

    return detect_faces
