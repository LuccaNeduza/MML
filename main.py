import argparse
import warnings 
from preprocess import data_generator
from train import train

warnings.filterwarnings('ignore')

"""
======== Pre-Processing/Training Code ========
Author: Luciana Menon
Code available on page: 'https://github.com/lucianamenon/ds-multimodal-emotion-recognition'

======== Cross-Attentional AV Fusion Model ========
Paper: A Joint Cross-Attention Model for Audio-Visual Fusion in Dimensional
Emotion Recognition
Authors: R Gnana Praveen, Wheidima Carneiro de Melo, Nasib Ullah, Haseeb Aslam1, Osama Zeeshan1, Théo
Denorme, Marco Pedersoli, Alessandro L. Koerich, Simon Bacon, Patrick Cardinal, and Eric Granger
Code available on page: 'https://github.com/praveena2j/Cross-Attentional-AV-Fusion'
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    preprocess = subparser.add_parser('preprocess', help='data pre-processing')
    training = subparser.add_parser('train', help='train models')

    training.add_argument('--seq_len', type=int, default=50)
    training.add_argument('--return_sequences', action='store_true', default=False)
    training.add_argument('--models', type=str, choices=['all', 'audio', 'video'], default='all')
    training.add_argument('--attention', type=str, choices=['all', 'cam'], default='cam')

    args = parser.parse_args()
    if args.command == 'preprocess':
        data_generator.data_generation()

    if args.command == 'train':
        seq_len = args.seq_len
        return_sequences = args.return_sequences
        model = args.models
        train.run(seq_len, return_sequences, model)