import tensorflow as tf
import argparse
import logging
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from vocab import Vocab
from dataset import Dataset

def parse_args():
    parser = argparse.ArgumentParser('mode settings')
    #parser.add_argument('--prepare', action='store_true',
    #                    help='build vocab')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--predict', action='store_true',
                        help='predict the most related para')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    
    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--hidden_size', type=int, default=200,
                                help='the hidden size')
    train_settings.add_argument('--embed_size', type=int, default=300,
                                help='word embedding size')
    train_settings.add_argument('--filters_num', type=int, default=4000,
                                help='number of CNN filters')
    train_settings.add_argument('--learning_rate', type=float, default=0.01,
                                help='the initial learning rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='batch size')
    train_settings.add_argument('--epochs', type=int, default=50,
                                help='train epochs')
    
    data_settings = parser.add_argument_group('data settings')
    data_settings.add_argument('--max_line', type=int, default=80000,
                               help='max line number for each file when reading train files and dev files')
    data_settings.add_argument('--p_num', type=int, default=20,
                               help='passage num for every document')
    data_settings.add_argument('--p_len', type=int, default=100,
                               help='paragraph length')
    data_settings.add_argument('--q_len', type=int, default=10,
                               help='question length')
    

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', type=str, 
                               default='../data/raw/trainset/search.train2.json',
                               help='train files')
    path_settings.add_argument('--dev_files', type=str, 
                               default='../data/raw/devset/search.dev2.json '+
                                       '../data/raw/devset/zhidao.dev2.json',
                               help='dev files')                          
    path_settings.add_argument('--test_files', type=str, 
                               default='../data/raw/testset/search.test2.json '+
                                       '../data/raw/testset/zhidao.test2.json',
                               help='test files')
    path_settings.add_argument('--save_dir', type=str, 
                               default='../data/save',
                               help='dir to put the predict results')
    path_settings.add_argument('--log_path', type=str,
                               default='../data/logs/default.log',
                               help='dir to save log file')
    return parser.parse_args()

def prepare(args):
    logger = logging.getLogger('brc')
    logger.info('Start preparing')
    vocab = Vocab(args)
    dataset = Dataset(args, vocab)
    #logger.info('Saving vocab...')
    #with open(os.path.join(args.save_dir, 'vocab.data'), 'wb') as fout:
    #    pickle.dump(vocab, fout)
    logger.info('Done with preparing')
    return dataset, vocab

def train(args, dataset, vocab):
    pass

def predict(args, dataset, vocab):
    pass

def run():
    args = parse_args()

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        print('no log path found')
        return
    
    logger.info('Running with args : {}'.format(args))

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    dataset, vocab = prepare(args)
    if args.train:
        train(args, dataset, vocab)
    if args.predict:
        predict(args, dataset, vocab)

if __name__ == '__main__':
    run()