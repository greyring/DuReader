import json
from vocab import Vocab
import numpy as np
import logging

class Dataset(object):
    def __init__(self, args, vocab):
        self.train_files = args.train_files
        self.dev_files = args.dev_files
        self.test_files = args.test_files
        self.max_line = args.max_line
        self.train = args.train
        self.predict = args.predict
        self.p_num = args.p_num
        self.p_len = args.p_len
        self.q_len = args.q_len

        self.train_set = []
        self.dev_set = []
        self.test_set= []
        self.logger = logging.getLogger("brc")
        self.vocab = vocab

        for file_name in self.train_files.split():
            self.logger.info('Reading train set...')
            self.train_set += self._load_dataset('train', file_name, self.max_line)
            self.logger.info('Train set size: %d docs.'%(len(self.train_set)))
        for file_name in self.dev_files.split():
            self.logger.info('Reading dev set...')
            self.dev_set += self._load_dataset('dev', file_name)
            self.logger.info('Dev set size: %d docs.'%(len(self.dev_set)))
        for file_name in self.test_files.split():
            self.logger.info('Reading test set...')
            self.test_set += self._load_dataset('test', file_name)
            self.logger.info('Test set size: %d docs.'%(len(self.test_set)))
       
    def add_words(self):
        self.logger.info('Adding word to Vocab model...')
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            for sample in data_set:
                self.vocab.add_list(sample['question'])
                for para in sample['paras']:
                    self.vocab.add_list(para)
        self.vocab.gen_ids()
        self.logger.info('Vocab has %d different words before filter'%(self.vocab.size()))
        self.vocab.filter_tokens_by_cnt(min_cnt=2)
        self.logger.info('Generate ids...')
        self.vocab.gen_ids()
        self.logger.info('Generate finishes')
        self.logger.info('Vocab has %d different words'%(self.vocab.size()))
        
    def convert_to_ids(self):
        self.logger.info('Converting words to ids...')
        self._conver_to_ids(self.vocab)
        self.logger.info('Convert finish')

    def _load_dataset(self, set_name, file_name, max_line=-1):
        dataset = []
        with open(file_name,'r') as fin:
            for lidx, line in enumerate(fin, 1):
                if max_line >= 0 and lidx > max_line: break
                sample = json.loads(line)
                data_sample = []
                for doc_id, doc in enumerate(sample['documents']):
                    if set_name == 'train' or set_name == 'dev':
                        data_sample.append({'question':sample['segmented_question'],
                                            'paras': doc['segmented_paragraphs'], 
                                            'answer': doc['most_related_para'],
                                            'question_id':sample['question_id'],
                                            'doc_id':doc_id})
                    else:
                        data_sample.append({'question':sample['segmented_question'],
                                            'paras':doc['segmented_paragraphs'],
                                            'question_id':sample['question_id'],
                                            'doc_id':doc_id})
                dataset += data_sample
        return dataset

    def _conver_to_ids(self, vocab):
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_ids'] = vocab.convert_to_ids(sample['question'])
                sample['paras_ids'] = [vocab.convert_to_ids(para) for para in sample['paras']]

    def _formal_q(self, question_ids, pad_id):
        #trunc or pad the question to q_len
        q_len = min(self.q_len, len(question_ids))
        q = []
        for idx in range(self.q_len):
            if idx<q_len:
                q.append(question_ids[idx])
            else:
                q.append(pad_id)
        return q, q_len

    def _formal_paras(self, paras_ids, pad_id):
        #trunc or pad paragraphs in paras to p_num
        #trunc or pad each para in paras to p_len
        paras = []
        paras_lens = []
        for para_id in range(self.p_num):
            if (para_id >= len(paras_ids)):
                paras.append([pad_id for idx in range(self.p_len)])
                paras_lens.append(0)
            else:
                para_len = min(self.p_len, len(paras_ids[para_id]))
                para = []
                for idx in range(self.p_len):
                    if idx<para_len:
                        para.append(paras_ids[para_id][idx])
                    else:
                        para.append(pad_id)
                paras.append(para)
                paras_lens.append(para_len)
        return paras, paras_lens

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)
    
    def _one_mini_batch(self, data, batch_indices, pad_id):
        batch_data = {'question':[],'question_len':[],'paras':[],'paras_len':[],'answer':[], 'question_id':[],'doc_id':[]}
        for i in batch_indices:
            one_case = data[i]
            q, q_len = self._formal_q(one_case['question_ids'], pad_id)
            batch_data['question'].append(q)
            batch_data['question_len'].append(q_len)
            paras, paras_lens = self._formal_paras(one_case['paras_ids'], pad_id)
            batch_data['paras'].append(paras)
            batch_data['paras_len'].append(paras_lens)
            if 'answer' in one_case:
                batch_data['answer'].append(one_case['answer'])
            else:
                batch_data['answer'].append(0)

            batch_data['question_id'].append(one_case['question_id'])
            batch_data['doc_id'].append(one_case['doc_id']) 
        return batch_data
