# -*- coding: utf-8 -*-
import api
import models
import os
import shutil
from collections import OrderedDict

try:
    import cPickle as pickle
except ImportError:
    import pickle

import click
import pims
import dlib
from tqdm import tqdm


def cluster(video_file_path, faces_output_path, skipped_duration, ending_time, step_duration, frame_scale_rate,
            gpu_memory_fraction, model):
    if model == 'mtcnn':
        face_detector = api.mtcnn_face_detector(gpu_memory_fraction)
    else:
        face_detector = api.dlib_cnn_face_detector(models.cnn_face_detector_model_location())

    sp = dlib.shape_predictor(models.pose_predictor_model_location())
    face_rec = dlib.face_recognition_model_v1(models.face_recognition_model_location())

    # 使用 pims + pyav 来读取视频，不会将视频加载到内存中，避免出现处理体积较大的视频时内存不足的情况
    vid = pims.Video(video_file_path)

    if skipped_duration is None or skipped_duration <= 0:
        skipped_frames_count = 0
    else:
        skipped_frames_count = int(vid.frame_rate * skipped_duration)

    if ending_time is None or ending_time <= 0:
        ending_frames_num = len(vid)
    else:
        ending_frames_num = min(int(vid.frame_rate * ending_time), len(vid))

    if step_duration is None or step_duration <= 0:
        step_frames_count = 1
    else:
        step_frames_count = int(vid.frame_rate * step_duration)

    face_to_descriptor = OrderedDict()

    for frame_num in tqdm(range(skipped_frames_count, ending_frames_num, step_frames_count)):
        frame = vid[frame_num]
        if frame_scale_rate is not None:
            frame = dlib.resize_image(frame, scale=frame_scale_rate)

        dets = face_detector(frame, 1)
        for i, rect in enumerate(dets):
            shape = sp(frame, rect)
            face_descriptor = face_rec.compute_face_descriptor(frame, shape, 1)
            #
            face_id = '%d_%d' % (frame_num, i)
            face_to_descriptor[face_id] = face_descriptor
            #
            file_path = os.path.join(faces_output_path, face_id)
            dlib.save_face_chip(frame, shape, file_path, size=150, padding=0.25)

    labels = dlib.chinese_whispers_clustering(list(face_to_descriptor.values()), 0.45)
    class_labels = set(labels)

    with open(os.path.join(faces_output_path, 'face_to_descriptor.pickle'), 'wb') as outfile:
        pickle.dump(face_to_descriptor, outfile)

    for class_label in class_labels:
        label_folder_path = os.path.join(faces_output_path, str(class_label))
        if not os.path.isdir(label_folder_path):
            os.makedirs(label_folder_path)

    for i, face_id in enumerate(face_to_descriptor.keys()):
        filename = '%s.jpg' % face_id
        file_path = os.path.join(faces_output_path, filename)
        dst_path = os.path.join(faces_output_path, str(labels[i]), filename)
        shutil.move(file_path, dst_path)


@click.command()
@click.argument('video_file_path')
@click.argument('faces_output_path')
@click.option('--skipped_duration', default=None, type=int, help='seconds')
@click.option('--ending_time', default=None, type=int, help='seconds')
@click.option('--step_duration', default=None, type=float, help='seconds')
@click.option('--frame_scale_rate', default=None, type=float, help='for reducing memory usage')
@click.option('--gpu_memory_fraction', default=0.5, type=float)
@click.option('--model', default=None, type=str)
def main(video_file_path, faces_output_path, skipped_duration, ending_time, step_duration, frame_scale_rate,
         gpu_memory_fraction, model):
    if not os.path.isfile(video_file_path):
        print('\"%s\" not found' % video_file_path)
        return
    if not os.path.isdir(faces_output_path):
        print('\"%s\" not exists' % faces_output_path)
        return

    cluster(video_file_path, faces_output_path, skipped_duration, ending_time, step_duration, frame_scale_rate,
            gpu_memory_fraction, model)


if __name__ == '__main__':
    main()
