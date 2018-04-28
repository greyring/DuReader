import json
from vocab import Vocab
import numpy as np
import logging

class Dataset(object):
    def __init__(self, args, vocab:Vocab):
        self.train_files = args.train_files
        self.dev_files = args.dev_files
        self.test_files = args.test_files
        self.max_line = args.max_line
        self.train = args.train
        self.dev = args.dev
        self.predict = args.predict
        self.p_num = args.p_num
        self.p_len = args.p_len
        self.q_len = args.q_len

        self.train_set = []
        self.dev_set = []
        self.test_set= []
        self.logger = logging.getLogger("brc")

        for file_name in self.train_files:
            self.logger.info('Reading train set...')
            self.train_set += self._load_dataset(file_name, self.max_line)
            self.logger.info('Train set size: %d docs.'%(len(self.train_set)))
        for file_name in self.dev_files:
            self.logger.info('Reading dev set...')
            self.dev_set += self._load_dataset(file_name)
            self.logger.info('Dev set size: %d docs.'%(len(self.dev_set)))
        for file_name in self.test_files:
            self.logger.info('Reading test set...')
            self.test_set += self._load_dataset(file_name)
            self.logger.info('Test set size: %d docs.'%(len(self.test_set)))
        
        self.logger.info('Adding word to Vocab model...')
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            vocab.add_list(data_set['question'])
            for para in data_set['paras']:
                vocab.add_list(para)
        vocab.filter_tokens_by_cnt(min_cnt=5)
        vocab.randomly_init_embeddings()
        self.logger.info('Vocab has %d different words'%(vocab.size()))
        
        self.logger.info('Converting words to ids...')
        self._conver_to_ids(vocab)
        self.logger.info('Convert finish')

    def _load_dataset(self, file_name:str, max_line=-1)->list:
        dataset = []
        with open(file_name,'r',encoding='utf-8') as fin:
            for lidx, line in enumerate(fin, 1):
                if max_line >= 0 and lidx > max_line: break
                sample = json.loads(line)
                data_sample = []
                for doc in sample['documents']:
                    if self.train or self.dev:
                        data_sample.append({'question':sample['segmented_question'],
                                            'paras': doc['segmented_paragraphs'], 
                                            'answer': doc['most_related_para']})
                    else:
                        data_sample.append({'question':sample['segmented_question'],
                                            'paras':doc['segmented_paragraphs']})
                dataset += data_sample
        return dataset

    def _conver_to_ids(self, vocab:Vocab):
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_ids'] = vocab.convert_to_ids(sample['question'])
                sample['paras_ids'] = [vocab.convert_to_ids(para) for para in sample['paras']]

    def _formal_q(self, question_ids:list, pad_id)->tuple:
        #trunc or pad the question to q_len
        q_len = min(self.q_len, len(question_ids))
        q = [(lambda idx: question_ids[idx] if idx<q_len else pad_id) for idx in range(self.q_len)]
        return q, q_len

    def _formal_paras(self, paras_ids:list, pad_id)->tuple:
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
                para = [(lambda idx: paras_ids[para_id][idx] if idx < para_len else pad_id) for idx in range(self.p_len)]
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
        batch_data = {'question':[],'question_len':[],'paras':[],'paras_len':[],'answer':[]}
        for i in batch_indices:
            one_case = data[i]
            q, q_len = self._formal_q(one_case['question_ids'], pad_id)
            batch_data['question'].append(q)
            batch_data['question_len'].append(q_len)
            paras, paras_lens = self._formal_paras(one_case['paras_ids'], pad_id)
            batch_data['paras'].append(paras)
            batch_data['paras_len'].append(paras_lens)
            if 'answer' in one_case:
                batch_data['answer'] = one_case['answer']
            else:
                batch_data['answer'] = -1
        return batch_data
