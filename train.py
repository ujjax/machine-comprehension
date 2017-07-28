from __future__ import print_function

import tensorflow as tf
import numpy as np
from model import Model
from data_stream import *


# Model Hyperparameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("val_percentage", 0.1, "Validation Percentage(default: 0.1)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{} = {}".format(attr.upper(), value))
print("")

with tf.Graph().as_default():

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	sess = tf.Session(config = config)

	with sess.as_default():

		model = Model()

		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-4)
		grads_and_vars = optimizer.compute_gradients(model.loss)

		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()

		grad_summaries = []
		for g, v in grads_and_vars:
			if g is not None:
				grad_hist_summary =  tf.summary.histogram("{}/grad/hist".format(v.name), g)
				sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
				grad_summaries.append(grad_hist_summary)
				grad_summaries.append(sparsity_summary)
		grad_summaries_merged = tf.summary.merge(grad_summaries)
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
		print("Writing to {}\n".format(out_dir))
		
		loss_summary = tf.summary.scalar("loss", model.loss)
		acc_summary = tf.summary.scalar("accuracy", model.accuracy)

		train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer =  tf.summary.FileWriter(train_summary_dir, sess.graph)

		dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
		dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
		dev_summary_writer =  tf.summary.FileWriter(dev_summary_dir, sess.graph)

		def train_step(q_batch, p_batch, start_batch, stop_batch):
			feed_dict = {
			  model.question_repres : q_batch,
			  model.passage_repres : p_batch,
			  model.start_index : start_batch,
			  model.stop_index : stop_batch
			}
			_, step, summaries, loss, accuracy = sess.run(
				[train_op, global_step, train_summary_op,model.loss, model.accuracy],
				feed_dict)
			time_str = datetime.datetime.now().isoformat()
#			print("{}: step {},loss {:g}, acc {:g}".format(time_str, step,loss, accuracy))
			train_summary_writer.add_summary(summaries, step)
		
		def val_step(writer=None):
			v = data_processing.get_batches()
			length = len()
			acc = []
			losses =[]
 #           print("Number of batches in dev set is " + str(length))
			for i in range(length):
				try:
					x_batch_dev, y_batch_dev = next(v)

					feed_dict = {
					  model.input_x: x_batch_dev,
					  model.input_y: one_hot(y_batch_dev),
					  model.dropout_keep_prob: 1.0
					}
					step,summaries, loss, accuracy = sess.run(
						[global_step, dev_summary_op, model.loss, model.accuracy],
						feed_dict)
					acc.append(accuracy)
					losses.append(loss)
					time_str = datetime.datetime.now().isoformat()
#                   print("batch " + str(i + 1) + " in dev >>" +
#                          " {}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))
					if writer:
						writer.add_summary(summaries, step)
				except StopIteration:
					pass
			print("\nMean accuracy=" + str(sum(acc)/len(acc)))
			print("Mean loss=" + str(sum(losses)/len(losses)))


		
		num_epoch = FLAGS.num_epochs
		g = get_batches()
		i=1
		print("Epoch >> 1")
		for epoch in range(num_epoch*89000):
#               x_batch = batch[0]
#               y_batch = batch[1]
			try: 
				q_batch, p_batch, start_batch, stop_batch = next(g)
				train_step(q_batch, p_batch, start_batch, stop_batch)
			except StopIteration:
				saver.save(sess, '/home/ujjawal/.ckpt')
				i+=1
				g = get_batches()
				val_step(writer = dev_summary_writer)
				print('Epoch >> '+ str(i))
				pass


"""
def main(_):
	print('Parameters:')
	print(FLAGS)

	train_path = FLAGS.train_path
    dev_path = FLAGS.dev_path
    test_path = FLAGS.test_path
    word_vec_path = FLAGS.word_vec_path
    log_dir = FLAGS.model_dir

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, help='Path to the train set.')
    parser.add_argument('--dev_path', type=str, help='Path to the dev set.')
    parser.add_argument('--test_path', type=str, help='Path to the test set.')
    parser.add_argument('--word_vec_path', type=str, help='Path the to pre-trained word vector model.')
    parser.add_argument('--model_dir', type=str, help='Directory to save model files.')
    parser.add_argument('--batch_size', type=int, default=60, help='Number of instances in each batch.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--lambda_l2', type=float, default=0.0, help='The coefficient of L2 regularizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout ratio.')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs for training.')
    parser.add_argument('--optimize_type', type=str, default='adam', help='Optimizer type.')
    parser.add_argument('--char_emb_dim', type=int, default=20, help='Number of dimension for character embeddings.')
    parser.add_argument('--char_lstm_dim', type=int, default=100, help='Number of dimension for character-composed embeddings.')
    parser.add_argument('--context_lstm_dim', type=int, default=100, help='Number of dimension for context representation layer.')
    parser.add_argument('--aggregation_lstm_dim', type=int, default=100, help='Number of dimension for aggregation layer.')
    parser.add_argument('--MP_dim', type=int, default=10, help='Number of perspectives for matching vectors.')
    parser.add_argument('--max_char_per_word', type=int, default=10, help='Maximum number of characters for each word.')
    parser.add_argument('--max_sent_length', type=int, default=100, help='Maximum number of words within each sentence.')
    parser.add_argument('--aggregation_layer_num', type=int, default=1, help='Number of LSTM layers for aggregation layer.')
    parser.add_argument('--context_layer_num', type=int, default=1, help='Number of LSTM layers for context representation layer.')
    parser.add_argument('--highway_layer_num', type=int, default=1, help='Number of highway layers.')
    parser.add_argument('--suffix', type=str, default='normal', required=True, help='Suffix of the model name.')
    parser.add_argument('--fix_word_vec', default=False, help='Fix pre-trained word embeddings during training.', action='store_true')
    parser.add_argument('--with_highway', default=False, help='Utilize highway layers.', action='store_true')
    parser.add_argument('--with_filter_layer', default=False, help='Utilize filter layer.', action='store_true')
    parser.add_argument('--word_level_MP_dim', type=int, default=-1, help='Number of perspectives for word-level matching.')
    parser.add_argument('--with_match_highway', default=False, help='Utilize highway layers for matching layer.', action='store_true')
    parser.add_argument('--with_aggregation_highway', default=False, help='Utilize highway layers for aggregation layer.', action='store_true')
    parser.add_argument('--with_lex_decomposition', default=False, help='Utilize lexical decomposition features.', action='store_true')
    parser.add_argument('--lex_decompsition_dim', type=int, default=-1, help='Number of dimension for lexical decomposition features.')
    parser.add_argument('--with_POS', default=False, help='Utilize POS information.', action='store_true')
    parser.add_argument('--with_NER', default=False, help='Utilize NER information.', action='store_true')
    parser.add_argument('--POS_dim', type=int, default=20, help='Number of dimension for POS embeddings.')
    parser.add_argument('--NER_dim', type=int, default=20, help='Number of dimension for NER embeddings.')
    parser.add_argument('--wo_left_match', default=False, help='Without left to right matching.', action='store_true')
    parser.add_argument('--wo_right_match', default=False, help='Without right to left matching', action='store_true')
    parser.add_argument('--wo_full_match', default=False, help='Without full matching.', action='store_true')
    parser.add_argument('--wo_maxpool_match', default=False, help='Without maxpooling matching', action='store_true')
    parser.add_argument('--wo_attentive_match', default=False, help='Without attentive matching', action='store_true')
    parser.add_argument('--wo_max_attentive_match', default=False, help='Without max attentive matching.', action='store_true')
    parser.add_argument('--wo_char', default=False, help='Without character-composed embeddings.', action='store_true')

#     print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    sys.stdout.flush()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
"""
