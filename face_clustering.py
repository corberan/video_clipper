# -*- coding: utf-8 -*-
import api
import models
import os
from glob import glob
import shutil
from collections import OrderedDict
from math import ceil

try:
    import cPickle as pickle
except ImportError:
    import pickle

import click
import dlib
from tqdm import tqdm


def get_frames(images_path_list, frame_scale_rate):
    for f in images_path_list:
        frame = dlib.load_rgb_image(f)
        if frame_scale_rate is not None:
            frame = dlib.resize_image(frame, scale=frame_scale_rate)
        yield frame


def cluster(faces_output_path, frame_scale_rate, iter_batch_size, gpu_memory_fraction, model):
    if model == 'mtcnn':
        face_detector = api.mtcnn_face_detector(gpu_memory_fraction)
    else:
        face_detector = api.dlib_cnn_face_detector(models.cnn_face_detector_model_location())

    sp = dlib.shape_predictor(models.pose_predictor_model_location())
    face_rec = dlib.face_recognition_model_v1(models.face_recognition_model_location())

    frame_face_descriptors = OrderedDict()

    image_file_suffix = '.jpg'
    image_file_suffix_len = len(image_file_suffix)
    images_path_list = [f for f in glob(os.path.join(faces_output_path, '?*{}'.format(image_file_suffix))) if
                        os.path.isfile(f)]
    images_size = len(images_path_list)

    with tqdm(total=images_size) as pbar:
        for i in range(ceil(images_size / iter_batch_size)):
            beginning_index = i * iter_batch_size
            ending_index = beginning_index + iter_batch_size
            if ending_index > images_size:
                images_path_batch = images_path_list[beginning_index:]
            else:
                images_path_batch = images_path_list[beginning_index:ending_index]
            frames = get_frames(images_path_batch, frame_scale_rate)
            for batch_index, (frame_dets, frame) in enumerate(face_detector(frames, 0)):
                face_detections = dlib.full_object_detections()
                frame_shape = frame.shape
                for det in frame_dets:
                    if type(det) == dlib.rectangle:
                        face_detections.append(sp(frame, det))
                    else:
                        rect = det.rect
                        new_left = max(rect.left() - round(rect.width() * 0.2), 0)
                        # new_top = max(rect.top() - round(rect.height() * 0.2), 0)
                        new_top = rect.top()
                        new_right = min(rect.right() + round(rect.width() * 0.2), frame_shape[1])
                        new_bottom = min(rect.bottom() + round(rect.height() * 0.3), frame_shape[0])
                        face_detections.append(sp(frame, dlib.rectangle(new_left, new_top, new_right, new_bottom)))

                face_detections_size = len(face_detections)
                frame_image_path = images_path_batch[batch_index]

                if face_detections_size > 0:
                    frame_num = int(os.path.basename(frame_image_path)[:-image_file_suffix_len])
                    face_descriptors = face_rec.compute_face_descriptor(frame, face_detections, 1)
                    frame_face_descriptors[frame_num] = face_descriptors

                    if face_detections_size == 1:
                        chip_file_path = os.path.join(os.path.dirname(frame_image_path), '%d_1' % frame_num)
                    else:
                        chip_file_path = os.path.join(os.path.dirname(frame_image_path), '%d' % frame_num)
                    dlib.save_face_chips(frame, face_detections, chip_file_path, size=150, padding=0)

                os.remove(frame_image_path)

            pbar.update(len(images_path_batch))

    face_descriptor_list = []
    for face_descriptors in frame_face_descriptors.values():
        for face_descriptor in face_descriptors:
            face_descriptor_list.append(face_descriptor)

    labels = dlib.chinese_whispers_clustering(face_descriptor_list, 0.45)
    class_labels = set(labels)

    with open(os.path.join(faces_output_path, 'frame_face_descriptors.pickle'), 'wb') as outfile:
        pickle.dump(frame_face_descriptors, outfile)

    unknown_label_name = 'unknown'
    for class_label in class_labels:
        label_folder_path = os.path.join(faces_output_path, '%s-%s' % (str(class_label), unknown_label_name))
        if not os.path.isdir(label_folder_path):
            os.makedirs(label_folder_path)

    label_index = 0
    for frame_num, face_descriptors in frame_face_descriptors.items():
        for i in range(len(face_descriptors)):
            filename = '%d_%d.jpg' % (frame_num, i + 1)
            src_path = os.path.join(faces_output_path, filename)
            dst_path = os.path.join(faces_output_path, '%s-%s' % (str(labels[label_index]), unknown_label_name),
                                    filename)
            shutil.move(src_path, dst_path)
            label_index += 1


@click.command()
@click.argument('faces_output_path')
@click.option('--frame_scale_rate', default=None, type=float)
@click.option('--iter_batch_size', default=1, type=int)
@click.option('--gpu_memory_fraction', default=0.5, type=float)
@click.option('--model', default=None, type=str)
def main(faces_output_path, frame_scale_rate, iter_batch_size, gpu_memory_fraction, model):
    if not os.path.isdir(faces_output_path):
        print('\"%s\" not exists' % faces_output_path)
        return

    cluster(faces_output_path, frame_scale_rate, iter_batch_size, gpu_memory_fraction, model)


if __name__ == '__main__':
    main()
