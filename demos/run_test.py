import time

import tensorflow as tf

import utils.data_utils as utils
from demos import style_transfer_tester
from utils.arg_parse_helpers import TestArgsParser


def main():
    arg_parser = TestArgsParser()
    args = arg_parser()

    # load content image
    content_image = utils.load_image(args.content, max_size=args.max_size)

    # open session
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    sess = tf.Session(config=soft_config)

    # build the graph
    transformer = style_transfer_tester.StyleTransferTester(
        session=sess,
        model_path=args.style_model,
        content_image=content_image,
    )

    # execute the graph
    start_time = time.time()
    output_image = transformer.test()
    end_time = time.time()

    # save result
    utils.save_image(output_image, args.output)

    # report execution time
    shape = content_image.shape
    print(
        'Execution time for a %d x %d image : %.2f ms' %
        (shape[2], shape[1], 1000 * (end_time - start_time))
    )


if __name__ == '__main__':
    main()
