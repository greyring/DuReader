import tensorflow as tf
import argparse
import logging
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from vocab import Vocab
from dataset import Dataset
from model import Model
import json
import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")

def parse_args():
    parser = argparse.ArgumentParser('mode settings')
    parser.add_argument('--prepare', action='store_true',
                        help='build vocab')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--predict', action='store_true',
                        help='predict the most related para')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    
    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--hidden_size', type=int, default=200,
                                help='the hidden size')
    train_settings.add_argument('--embed_size', type=int, default=200,
                                help='word embedding size')
    train_settings.add_argument('--filters_num', type=int, default=4000,
                                help='number of CNN filters')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='the initial learning rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='batch size')
    #train_settings.add_argument('--dropout_keep_prob', type=float, default=1.0,
    #                            help='dropout keep probability')
    train_settings.add_argument('--epoches', type=int, default=50,
                                help='train epoches')
    
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
                               default='../data/demo/trainset/search.train.json',
                               help='train files')
    path_settings.add_argument('--dev_files', type=str, 
                               default='../data/demo/devset/search.dev.json',
                               help='dev files')                          
    path_settings.add_argument('--test_files', type=str, 
                               default='../data/demo/testset/search.test.json',
                               help='test files')
    path_settings.add_argument('--save_dir', type=str, 
                               default='../data/save',
                               help='dir to put the predict results')
    #path_settings.add_argument('--log_path', type=str,
    #                           default='../data/logs/default.log',
    #                           help='dir to save log file')
    return parser.parse_args()

def prepare(args):
    logger = logging.getLogger('brc')
    logger.info('Start preparing')
    vocab = Vocab(args)
    dataset = Dataset(args, vocab)
    dataset.add_words()

    logger.info('Initing embedding...')
    vocab.randomly_init_embeddings()
    logger.info('Done with init')

    logger.info('Saving vocab...')
    with open(os.path.join(args.save_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)
    logger.info('Done with preparing')

    del vocab
    del dataset
    return 

def train(args):
    logger = logging.getLogger('brc')

    logger.info('Training start...')
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.save_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    logger.info('Done with loading')

    dataset = Dataset(args, vocab)
    dataset.convert_to_ids()
    
    model = Model(args, dataset, vocab)
    model.train()
    del vocab
    del dataset
    del model
    logger.info('Done with training')
    return

def predict(args):
    logger = logging.getLogger('brc')

    logger.info('Predicting start...')
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.save_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    logger.info('Done with loading')
    
    dataset = Dataset(args,vocab)
    dataset.convert_to_ids()

    model = Model(args, dataset, vocab)
    model.restore()
    model.predict()

    logger.info('Start changing...')
    answers = {}
    with open(os.path.join(args.save_dir, 'result.json'), 'r') as fin:
        for line in fin:
            d = json.loads(line)
            answers[(d['question_id'], d['doc_id'])] = d['most_related_para']
    
    bad_num = 0
    for test_file in args.test_files.split():
        with open(test_file, 'r') as fin:
            with open(test_file+'.json', 'w') as fout:
                for line in fin:
                    sample = json.loads(line)
                    for docid, doc in enumerate(sample['documents']):
                        if ((sample['question_id'], docid) in answers) and \
                            (answers[(sample['question_id'], docid)] < len(doc['paragraphs'])):
                            doc['most_related_para'] = answers[(sample['question_id'], docid)]
                        else:
                            bad_num += 1
                    fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    
    logger.info('Finish changing with {} bad answers'.format(bad_num))

    del vocab
    del dataset
    del model
    logger.info('Done with predicting')
    return

def run():
    args = parse_args()

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(args.save_dir, 'default.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info('Running with args : {}'.format(args))

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.predict:
        predict(args)

if __name__ == '__main__':
    run()
