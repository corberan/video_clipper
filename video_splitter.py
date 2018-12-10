import os
import imageio
from tqdm import tqdm
import click


def split(video_file_path, faces_output_path, skipped_duration, ending_time, step_duration):
    reader = imageio.get_reader(video_file_path)

    meta_data = reader.get_meta_data()
    video_fps = meta_data['fps']
    video_frames_count = meta_data['nframes']

    if skipped_duration is None or skipped_duration <= 0:
        skipped_frames_count = 0
    else:
        skipped_frames_count = round(video_fps * skipped_duration)

    if ending_time is None or ending_time <= 0:
        ending_frames_num = video_frames_count
    else:
        ending_frames_num = min(round(video_fps * ending_time), video_frames_count)

    if step_duration is None or step_duration <= 0:
        step_frames_count = 1
    else:
        step_frames_count = round(video_fps * step_duration)

    for frame_num in tqdm(range(skipped_frames_count, ending_frames_num, step_frames_count)):
        reader.set_image_index(frame_num)
        frame = reader.get_next_data()
        imageio.imwrite(os.path.join(faces_output_path, ('%d.jpg' % frame_num)), frame)


@click.command()
@click.argument('video_file_path')
@click.argument('faces_output_path')
@click.option('--skipped_duration', default=None, type=int, help='seconds')
@click.option('--ending_time', default=None, type=int, help='seconds')
@click.option('--step_duration', default=None, type=float, help='seconds')
def main(video_file_path, faces_output_path, skipped_duration, ending_time, step_duration):
    if not os.path.isfile(video_file_path):
        print('\"%s\" not found' % video_file_path)
        return
    if not os.path.isdir(faces_output_path):
        print('\"%s\" not exists' % faces_output_path)
        return

    split(video_file_path, faces_output_path, skipped_duration, ending_time, step_duration)


if __name__ == '__main__':
    main()
