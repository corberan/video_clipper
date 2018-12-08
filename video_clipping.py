import os
from collections import OrderedDict
import glob
import pims
import click


def get_formatted_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return '%02d:%02d:%02d' % (h, m, s)
    elif m > 0:
        return '%02d:%02d' % (m, s)
    else:
        return '00:%02d' % s


def get_duration_str(start_frame_num, end_frame_num, frame_rate):
    return '{} - {}'.format(get_formatted_time(start_frame_num/frame_rate), get_formatted_time(end_frame_num/frame_rate))


def clip(video_file_path, face_classes_path, split_blank_duration):
    items = os.listdir(face_classes_path)
    if len(items) == 0:
        print('no items in path')
        return

    class_names = OrderedDict()
    for item in items:
        label_folder_path = os.path.join(face_classes_path, item)
        if os.path.isdir(label_folder_path):
            label_and_name = item.split('-', 1)
            if len(label_and_name) == 2:
                face_label = int(label_and_name[0])
                class_names[face_label] = label_and_name[1]

    print('please select:')
    class_names_list = list(set(class_names.values()))
    for i, name in enumerate(class_names_list):
        print('{}: {}'.format(i, name))

    user_input = input()
    selected_name = class_names_list[int(user_input)]
    print('you select {}'.format(selected_name))

    frame_nums = set()
    for label, name in class_names.items():
        if name == selected_name:
            label_folder_path = os.path.join(face_classes_path, '{}-{}'.format(label, name))
            for f in glob.glob(os.path.join(label_folder_path, '*.jpg')):
                if os.path.isfile(f):
                    image_filename = os.path.basename(f)
                    pos = image_filename.find('_')
                    if pos > 0:
                        frame_num = image_filename[:pos]
                        frame_nums.add(int(frame_num))

    frames_count = len(frame_nums)
    if frames_count == 0:
        print('no images found')
        return
    else:
        frame_nums_list = list(frame_nums)
        frame_nums_list.sort()

        vid = pims.Video(video_file_path)
        blank_frames_count = vid.frame_rate * split_blank_duration

        result = []
        section = [frame_nums_list[0]]
        for i in range(1, frames_count):
            frame_num = frame_nums_list[i]
            if frame_num - section[-1] < blank_frames_count:
                section.append(frame_num)
            else:
                result.append(get_duration_str(section[0], section[-1], vid.frame_rate))
                section = [frame_num]

        if len(result) == 0:
            result.append(get_duration_str(section[0], section[-1], vid.frame_rate))

        print(result)


@click.command()
@click.argument('video_file_path')
@click.argument('face_classes_path')
@click.argument('split_blank_duration', type=float)
def main(video_file_path, face_classes_path, split_blank_duration):
    if not os.path.isfile(video_file_path):
        print('\"%s\" not found' % video_file_path)
        return
    if not os.path.isdir(face_classes_path):
        print('\"%s\" not exists' % face_classes_path)
        return

    clip(video_file_path, face_classes_path, split_blank_duration)


if __name__ == '__main__':
    main()
