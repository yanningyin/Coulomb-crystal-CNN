#!/usr/bin/env python3

import argparse
import sys
import os
import ast


def parse_range_arg(arg_string):
    try:
        return eval(arg_string)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid range argument: {str(e)}")


class CLI:
    """Command line interface that allows the user to specify parameters"""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="Specify parameters")
        parser.add_argument('input_image_dir',
                            nargs='+',
                            default='images',
                            help='Directory name(s) of input images')
        parser.add_argument('--output_file_dir',
                            '-o',
                            type=str,
                            default='output',
                            help='Directory to save output files')
        parser.add_argument('--mode',
                            '-md',
                            choices=['train', 'test'],
                            default='train',
                            help='Mode for running the program: train or test')
        parser.add_argument('--test_model_path',
                            '-mp',
                            type=str,
                            default='',
                            help='Path to the model to be loaded for image classification')
        parser.add_argument('--label',
                            '-l',
                            choices=['N', 'n', 'T', 't', 'NT', 'nt'],
                            default='T',
                            help='Label(s) to be classified. N or n: number, T or t: temperature, NT or nt: both')
        parser.add_argument('--range_N',
                            '-rN',
                            type=parse_range_arg,
                            help='Range of the label N (number)')
        parser.add_argument('--range_T',
                            '-rT',
                            type=parse_range_arg,
                            help='Range of the label T (temperature)')
        parser.add_argument('--check_image',
                            '-ci',
                            action='store_true',
                            help='Check if image is valid (default: False)')
        parser.add_argument('--model',
                            '-m',
                            default='alexnet',
                            choices=['custom', 'resnet18', 'resnet50', 'resnet101', 'resnext101_32x8d', 'alexnet', 'mnasnet1_0', 'mnasnet1_3', 'googlenet', 'vgg16'],
                            help='Model for the task')
        parser.add_argument('--use_pretrained_weights',
                            '-w',
                            action='store_true',
                            help='Whether to use the pretrained weights')
        parser.add_argument('--convert_to_rgb',
                            '-rgb',
                            action='store_true',
                            help='Whether to convert the image from L to RGB (default: False)')

        parser.add_argument('--class_range_N',
                            '-crN',
                            type=parse_range_arg,
                            default=argparse.SUPPRESS,
                            help='The range of class for classification, default value is the same as range_N')

        parser.add_argument('--class_range_T',
                            '-crT',
                            type=parse_range_arg,
                            default=argparse.SUPPRESS,
                            help='The range of class for classification, default value is the same as range_T')

        self.args = parser.parse_args()

        if self.args.mode == 'test' and not self.args.test_model_path:
            parser.error('When mode is set to test, test_model_path must not be empty.')
        
        if not hasattr(self.args, 'class_range_N'):
            self.args.class_range_N = self.args.range_N

        if not hasattr(self.args, 'class_range_T'):
            self.args.class_range_T = self.args.range_T

        

def main():
    cli = CLI()


if __name__ == "__main__":
    main()
