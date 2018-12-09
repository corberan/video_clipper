# -*- coding: utf-8 -*-
import api
import models
import os
import platform
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
import cv2


# 在 Linux 下 pims 运行良好，内存占用仅 200 MB 左右，但在 Windows 上会不断分配内存直至无法分配最后报错
# 可能是新版的 PyAV 导致的，所以暂时在 Windows 上切换为 opencv 读取视频
# opencv 跳帧读取会消耗很大的 CPU 资源，没办法
def open_video(video_file_path):
    if platform.system() == "Windows":
        vid = cv2.VideoCapture(video_file_path)
        video_fps = vid.get(5)  # CV_CAP_PROP_FPS
        video_frames_count = int(vid.get(7))  # CV_CAP_PROP_FRAME_COUNT
    else:
        vid = pims.Video(video_file_path)
        video_fps = vid.frame_rate
        video_frames_count = len(vid)

    def get_frame(frame_num):
        if frame_num > video_frames_count:
            return None

        if platform.system() == "Windows":
            vid.set(1, frame_num)  # CV_CAP_PROP_POS_FRAMES
            ret, frame = vid.read()
            if not ret:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return vid[frame_num]

    return video_fps, video_frames_count, get_frame


def cluster(video_file_path, faces_output_path, skipped_duration, ending_time, step_duration, frame_scale_rate,
            gpu_memory_fraction, model):
    if model == 'mtcnn':
        face_detector = api.mtcnn_face_detector(gpu_memory_fraction)
    else:
        face_detector = api.dlib_cnn_face_detector(models.cnn_face_detector_model_location())

    sp = dlib.shape_predictor(models.pose_predictor_model_location())
    face_rec = dlib.face_recognition_model_v1(models.face_recognition_model_location())

    video_fps, video_frames_count, get_frame = open_video(video_file_path)

    if skipped_duration is None or skipped_duration <= 0:
        skipped_frames_count = 0
    else:
        skipped_frames_count = int(video_fps * skipped_duration)

    if ending_time is None or ending_time <= 0:
        ending_frames_num = video_frames_count
    else:
        ending_frames_num = min(int(video_fps * ending_time), video_frames_count)

    if step_duration is None or step_duration <= 0:
        step_frames_count = 1
    else:
        step_frames_count = int(video_fps * step_duration)

    face_to_descriptor = OrderedDict()

    for frame_num in tqdm(range(skipped_frames_count, ending_frames_num, step_frames_count)):
        frame = get_frame(frame_num)
        if frame is None:
            break

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
