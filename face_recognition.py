import api
import models
import os

try:
    import cPickle as pickle
except ImportError:
    import pickle

import dlib
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import click


def recognition(video_file_path, face_classifier_file_path, skipped_duration, frame_scale_rate, save_result,
                gpu_memory_fraction, model):
    if model == 'mtcnn':
        face_detector = api.mtcnn_face_detector(gpu_memory_fraction)
    else:
        face_detector = api.dlib_cnn_face_detector(models.cnn_face_detector_model_location())

    sp = dlib.shape_predictor(models.pose_predictor_model_location())
    face_rec = dlib.face_recognition_model_v1(models.face_recognition_model_location())

    font = ImageFont.truetype('‪C:\\Windows\\Fonts\\msyh.ttc', 14)  # 注意改成开源字体

    with open(face_classifier_file_path, 'rb') as infile:
        (model, class_names) = pickle.load(infile)

    vid = cv2.VideoCapture(video_file_path)

    video_writer = None
    if save_result:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_fps = vid.get(5)
        video_width = vid.get(3)
        video_height = vid.get(4)
        if frame_scale_rate is not None:
            video_width = int(video_width * frame_scale_rate)
            video_height = int(video_height * frame_scale_rate)

        video_writer = cv2.VideoWriter(
            os.path.join(os.path.dirname(face_classifier_file_path), 'face_recognition_result.mp4'), fourcc,
            video_fps, (video_width, video_height))

    if skipped_duration is not None:
        vid.set(0, skipped_duration * 1000)

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        if frame_scale_rate is not None:
            frame = cv2.resize(frame, None, fx=frame_scale_rate, fy=frame_scale_rate)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for batch_index, (frame_dets, _) in enumerate(face_detector([frame_rgb], 0)):
            dets_size = len(frame_dets)
            if dets_size > 0:
                face_descriptors = np.zeros((dets_size, 128))
                for i, det in enumerate(frame_dets):
                    if type(det) == dlib.rectangle:
                        shape = sp(frame_rgb, det)
                    else:
                        shape = sp(frame_rgb, det.rect)
                    face_descriptor = face_rec.compute_face_descriptor(frame_rgb, shape, 1)
                    face_descriptors[i:i + 1:] = np.asarray(face_descriptor)
                predict_labels = model.predict(face_descriptors)
                predictions = model.predict_proba(face_descriptors)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                img = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(img)

                for i, det in enumerate(frame_dets):
                    text = '{}: {:.2%}'.format(class_names[predict_labels[i]], best_class_probabilities[i])
                    if type(det) == dlib.rectangle:
                        rect = det
                    else:
                        rect = det.rect
                    draw.text((rect.left() + 2, rect.bottom() + 2), text, font=font, fill=(255, 0, 0))
                    draw.rectangle([(rect.left(), rect.top()), (rect.right(), rect.bottom())],
                                   outline=(255, 0, 0))

                frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        cv2.imshow(video_file_path, frame)
        if save_result:
            video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    if save_result:
        video_writer.release()
    cv2.destroyAllWindows()


@click.command()
@click.argument('video_file_path')
@click.argument('face_classifier_file_path')
@click.option('--skipped_duration', default=None, type=int, help='seconds')
@click.option('--frame_scale_rate', default=None, type=float, help='for reducing memory usage')
@click.option('--save_result', default=False, type=bool, help='save recognition result video')
@click.option('--gpu_memory_fraction', default=0.5, type=float)
@click.option('--model', default=None, type=str)
def main(video_file_path, face_classifier_file_path, skipped_duration, frame_scale_rate, save_result,
         gpu_memory_fraction, model):
    if not os.path.isfile(video_file_path):
        print('\"%s\" not found' % video_file_path)
        return
    if not os.path.isfile(face_classifier_file_path):
        print('\"%s\" not found' % face_classifier_file_path)
        return

    recognition(video_file_path, face_classifier_file_path, skipped_duration, frame_scale_rate, save_result,
                gpu_memory_fraction, model)


if __name__ == '__main__':
    main()
