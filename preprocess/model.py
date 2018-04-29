import tensorflow as tf
from tensorflow.python import debug as tfdbg
import tensorflow.contrib as tc
import numpy as np
import logging
import time
import json
import os
from vocab import Vocab
from dataset import Dataset

class Model(object):
    def __init__(self, args, dataset, vocab):
        self.logger = logging.getLogger('brc')
        self.vocab = vocab
        self.dataset = dataset

        self.epoches = args.epoches
        self.batch_size = args.batch_size
        self.q_len = args.q_len
        self.p_num = args.p_num
        self.p_len = args.p_len
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.filters_num = args.filters_num
        #self.dropout_keep_prob = args.dropout_keep_prob
        self.learning_rate = args.learning_rate
        self.save_dir = args.save_dir

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        #self.sess = tfdbg.LocalCLIDebugWrapperSession(self.sess)
        #self.sess.add_tensor_filter('has_inf_or_nan', tfdbg.has_inf_or_nan)
        
        self._build_graph()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
    
    def _build_graph(self):
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._HL()
        self._CNN()
        self._PT()
        self._compute_loss()
        self._create_train_op()
        self._summary()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in tf.trainable_variables()])
        self.logger.info('There are {} parameters in the model'.format(param_num))
        self.logger.info('%s'%(str([x.name for x in tf.global_variables()])))
    
    def _setup_placeholders(self):
        self.q = tf.placeholder(tf.int32, [None, self.q_len])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.p = tf.placeholder(tf.int32, [None, self.p_num, self.p_len])
        self.p_length = tf.placeholder(tf.int32, [None, self.p_num])
        self.answer = tf.placeholder(tf.int32, [None])
        #self.dropout_keep_prob = tf.placeholder(tf.float32)

        self.answer_oh = tf.one_hot(self.answer, self.p_num, on_value=1.0, off_value=0.0)

    def _embed(self):
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.embed_size),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True
            )
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)#batch_size, q_len, embed_size
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)#it's ok batch_size, p_num, p_len, embed_size

    def _HL(self):
        with tf.variable_scope('Hidden'):
            self.q_hid = tc.layers.fully_connected(self.q_emb, self.hidden_size, 
                                    activation_fn=tf.tanh,
                                    reuse=None) #batch_size, q_len, hidden_size
            self.p_hid = tc.layers.fully_connected(self.p_emb, self.hidden_size, 
                                    activation_fn=tf.tanh,
                                    reuse=True,
                                    scope='fully_connected') #batch_size, p_num, p_len, hidden_size
              
    def _CNN(self):
        with tf.variable_scope('CNN'):
            self.p_hid = tf.reshape(self.p_hid, [-1, self.p_len, self.hidden_size])#batch_size*p_num, p_len, hidden_size
            self.q_cnn = tf.layers.conv1d(self.q_hid, self.filters_num, kernel_size=5, name='CNNpq', reuse=None)#batch_size, q_len-4, filters_num
            self.p_cnn = tf.layers.conv1d(self.p_hid, self.filters_num, kernel_size=5, name='CNNpq', reuse=True)#batch_size*p_num, p_len-4, fileters_num 

    def _PT(self):
        with tf.variable_scope('PT'):
            self.q_out = tf.layers.max_pooling1d(self.q_cnn, pool_size=int(self.q_cnn.shape[1]), strides=1, name='Poolq')
            #batch_size, 1, filters_num
            self.p_out = tf.layers.max_pooling1d(self.p_cnn, pool_size=int(self.p_cnn.shape[1]), strides=1, name='Poolp')
            #batch_size*p_num, 1, filters_num
            self.p_out = tf.reshape(self.p_out, [tf.shape(self.q)[0], -1, self.filters_num])
            self.q_out = tf.tile(self.q_out, [1, self.p_num, 1])

            self.q_out = tf.tanh(self.q_out)#batch_size, p_num, filters_num
            self.p_out = tf.tanh(self.p_out)#batch_size, p_num, filters_num
            
            mean, var = tf.nn.moments(self.p_out, axes=[0])
            self.p_out = tf.nn.batch_normalization(self.p_out, mean, var, None, None, 0.001)
    
    def _compute_loss(self):
        with tf.variable_scope('loss'):
            #GESD
            self.sim = 1/(1 + tf.norm(self.p_out - self.q_out, axis=2))\
            * 1/(1 + tf.exp(-1.0 * tf.reduce_sum(self.p_out * self.q_out, axis=2)))#batch_size, p_num
            #self.sim = tf.reduce_sum(self.p_out * self.q_out, axis=2)/(tf.norm(self.p_out, axis=2) * tf.norm(self.q_out, axis=2))
            self.hinge_loss = tf.losses.hinge_loss(self.answer_oh, self.sim)
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = self.hinge_loss + 0.0001 * l2_loss

            self.predict_answer = tf.cast(tf.argmax(self.sim, axis=1), tf.int32)
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.answer, self.predict_answer), tf.float32))/tf.cast(tf.shape(self.q)[0], tf.float32)
    
    def _create_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -0.0001, 0.0001), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)
    
    def _summary(self):
        #self.q_embed_sum = tf.summary.scalar('q_embed', self.q_emb[0][0][0])
        #self.q_hid = tf.summary.scalar('q_hid', self.q_hid[0][0][0])
        #self.q_cnn = tf.summary.scalar('q_cnn', self.q_cnn[0][0][0])
        #self.q_out_sum  = tf.summary.scalar('q_out', self.q_out[0][0][0])
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        self.summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'), graph=self.sess.graph)

    def _train_epoch(self, base_step, train_batches):
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(train_batches, 1):
            feed_dict = {self.p: batch['paras'],
                         self.q: batch['question'],
                         self.p_length: batch['paras_len'],
                         self.q_length: batch['question_len'],
                         self.answer: batch['answer']}
            _, loss, train_summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict)
            total_loss += loss * len(batch['question'])
            total_num += len(batch['question'])
            n_batch_loss += loss
            self.summary_writer.add_summary(train_summary, base_step + bitx)
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def train(self):
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        best_accuracy = 0
        for epoch in range(1, self.epoches + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = self.dataset.gen_mini_batches('train', self.batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch((epoch-1) * self.batch_size, train_batches)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            self.logger.info('Evaluating the model after epoch {}'.format(epoch))
            eval_batches = self.dataset.gen_mini_batches('dev', self.batch_size, pad_id, shuffle=False)
            eval_loss, eval_accuracy = self._evaluate(eval_batches)
            self.logger.info('Dev eval loss {}, accuracy {}'.format(eval_loss, eval_accuracy))
            
            if (eval_accuracy > best_accuracy):
                best_accuracy = eval_accuracy
                self.saver.save(self.sess, os.path.join(self.save_dir, 'model'))
                self.logger.info('Model saved at {}, with prefix {}.'.format(self.save_dir, 'model'))
    
    def _evaluate(self, eval_batches):
        total_num, total_loss, total_accuracy = 0, 0, 0
        for bitx, batch in enumerate(eval_batches, 1):
            feed_dict = {self.p: batch['paras'],
                         self.q: batch['question'],
                         self.p_length: batch['paras_len'],
                         self.q_length: batch['question_len'],
                         self.answer: batch['answer']}
            accuracy, loss = self.sess.run([self.accuracy, self.loss], feed_dict)
            total_loss += loss * len(batch['question'])
            total_accuracy += accuracy * len(batch['question'])
            total_num += len(batch['question'])
        return 1.0 * total_loss / total_num, 1.0 * total_accuracy /total_num
    
    def predict(self):
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        self.logger.info('Predicting the model')
        predict_batches = self.dataset.gen_mini_batches('test', self.batch_size, pad_id, shuffle=False)
        pred_answers=[]
        for bitx, batch in enumerate(predict_batches, 1):
            feed_dict = {self.p: batch['paras'],
                         self.q: batch['question'],
                         self.p_length: batch['paras_len'],
                         self.q_length: batch['question_len'],
                         self.answer: batch['answer']}
            predict_answer = self.sess.run([self.predict_answer], feed_dict)#batch_size 
            predict_answer = predict_answer[0]
            for idx in range(len(batch['question'])):
                pred_answers.append({'question_id':batch['question_id'][idx],
                                     'doc_id':batch['doc_id'][idx],
                                     'most_related_para':int(predict_answer[idx])})
        with open(os.path.join(self.save_dir, 'result.json'), 'w') as fout:
            for pred_answer in pred_answers:
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

        self.logger.info('Saving results to {}, with name {}'.format(self.save_dir, 'result.json'))
    
    def restore(self):
        self.saver.restore(self.sess, os.path.join(self.save_dir, 'model'))
        self.logger.info('Model restored from {}, with prefix {}'.format(self.save_dir, 'model'))
