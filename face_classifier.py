import os
from glob import glob
from collections import OrderedDict

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
from sklearn.svm import SVC
import click


def classify(faces_output_path):
    items = os.listdir(faces_output_path)
    if len(items) == 0:
        raise Exception('no items in the path')

    class_names = OrderedDict()

    frame_face_descriptors_file_name = 'frame_face_descriptors.pickle'
    if frame_face_descriptors_file_name not in items:
        raise Exception('file \"%s\" not found' % frame_face_descriptors_file_name)
    with open(os.path.join(faces_output_path, frame_face_descriptors_file_name), 'rb') as infile:
        frame_face_descriptors = pickle.load(infile)

    image_file_suffix = '.jpg'
    descriptors = []
    labels = []

    for item in items:
        label_folder_path = os.path.join(faces_output_path, item)
        if os.path.isdir(label_folder_path):
            label_and_name = item.split('-', 1)
            if len(label_and_name) == 2:
                face_label = int(label_and_name[0])
                class_names[face_label] = label_and_name[1]
                #
                for f in glob(os.path.join(label_folder_path, '???*{}'.format(image_file_suffix))):
                    if os.path.isfile(f):
                        image_filename = os.path.basename(f)
                        pos = image_filename.find('_')
                        if pos > 0:
                            frame_num = int(image_filename[:pos])
                            face_descriptors = frame_face_descriptors[frame_num]
                            for face_descriptor in face_descriptors:
                                descriptors.append(np.asarray(face_descriptor))
                                labels.append(face_label)

    model = SVC(kernel='linear', probability=True)
    model.fit(descriptors, labels)

    with open(os.path.join(faces_output_path, 'face_classifier.pickle'), 'wb') as outfile:
        pickle.dump((model, class_names), outfile)


@click.command()
@click.argument('faces_output_path')
def main(faces_output_path):
    if not os.path.isdir(faces_output_path):
        print('\"%s\" not exists' % faces_output_path)
        return

    classify(faces_output_path)


if __name__ == '__main__':
    main()
