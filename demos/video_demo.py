from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
from tqdm import tqdm
import cv2
from utils.demo_utils import StyleTransferDemo
import numpy as np
from typing import List
import click


def get_frames(video_path: str) -> List[np.ndarray]:
    """
    Load frames from video.
    :param video_path: path to video.
    :return: loaded frames.
    """

    video_reader = FFMPEG_VideoReader(video_path)
    frames = []

    for _ in tqdm(range(video_reader.nframes),
                  desc='Getting video frames'):
        frames.append(video_reader.read_frame())

    return frames


def generate_video_by_frames(path: str, frames: List[np.ndarray]):
    """
    Generate video file by frames sequence.
    :param path: path where to store resulting video.
    :param frames: frames sequence.
    """

    (height, width, _) = frames[0].shape
    video = cv2.VideoWriter(path, -1, 30, (width, height))

    for image in tqdm(frames, desc='Writing video'):
        video.write(image)

    video.release()


def combine_frames(left_frames: List[np.ndarray],
                   right_frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Combine two sequences of frames into one by concatenating them.
    :param left_frames: left side sequence.
    :param right_frames: right side sequence.
    :return: concatenated sequence.
    """

    if len(left_frames) != len(right_frames):
        raise ValueError('Sequences of frames must be same length!')

    combined_frames = []
    for left_frame, right_frame in zip(left_frames, right_frames):
        combined_frame = np.concatenate([left_frame, right_frame], axis=1)
        combined_frames.append(combined_frame)

    return combined_frames


@click.command()
@click.option('--video_path',
              help='Path to video that need to process.',
              default='../data/videos/Africa.mp4')
@click.option('--result_path',
              help='Path to file where to store results.',
              default='../data/videos/Africa_styled.mp4')
@click.option('--model_path',
              help='Path to model protobuf.',
              default='../model/optimized_model.pb')
@click.option('--image_size',
              help='Output image size.',
              default='360,640')
@click.option('--batch_size',
              help='Batch size.',
              default='1')
def video_demo(video_path: str,
               result_path: str,
               model_path: str,
               image_size: str,
               batch_size: str):
    image_size = [int(size) for size in image_size.split(',')]
    batch_size = int(batch_size)

    transformer = StyleTransferDemo(
        model_path=model_path,
        input_shape=image_size,
        scope='style_transfer_cnn'
    )

    original_frames = get_frames(video_path=video_path)
    original_frames = [
        cv2.resize(frame, dsize=(image_size[1], image_size[0]))
        for frame in original_frames
    ]

    counter = tqdm(original_frames, desc='Processing frames')
    num_frames = len(original_frames)
    num_batches = num_frames // batch_size
    num_batches += int(num_batches % batch_size != 0)
    styled_frames = []

    for i in range(num_batches):
        begin = i * batch_size
        end = min((i + 1) * batch_size, num_frames)

        curr_frames = np.array(original_frames[begin:end])
        out_frames = transformer(curr_frames)

        if batch_size != 1:
            styled_frames.extend(out_frames)
        else:
            styled_frames.append(out_frames)

        counter.update(n=(end - begin))

    resulting_images = combine_frames(
        left_frames=original_frames,
        right_frames=styled_frames
    )

    generate_video_by_frames(result_path, frames=resulting_images)


if __name__ == '__main__':
    video_demo()
