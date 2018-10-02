import numpy as np
import os
import tensorflow as tf
import utility as Utility
import argparse
from model_BiGAN import BiGAN as Model
from make_datasets_MNIST import Make_datasets_MNIST as Make_datasets

def parser():
    parser = argparse.ArgumentParser(description='train LSGAN')
    parser.add_argument('--batch_size', '-b', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--log_file_name', '-lf', type=str, default='log180926', help='log file name')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='epoch')
    parser.add_argument('--file_name', '-fn', type=str, default='./mnist.npz', help='file name of data')
    parser.add_argument('--valid_span', '-vs', type=int, default=50, help='validation span')

    return parser.parse_args()

args = parser()


#global variants
BATCH_SIZE = args.batch_size
LOGFILE_NAME = args.log_file_name
EPOCH = args.epoch
FILE_NAME = args.file_name
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNEL = 1
BASE_CHANNEL = 32
NOISE_UNIT_NUM = 200
NOISE_MEAN = 0.0
NOISE_STDDEV = 1.0
TEST_DATA_SAMPLE = 5 * 5
L2_NORM = 0.001
KEEP_PROB_RATE = 0.5
SEED = 1234
SCORE_ALPHA = 0.9 # using for cost function
VALID_SPAN = args.valid_span
np.random.seed(seed=SEED)
BOARD_DIR_NAME = './tensorboard/' + LOGFILE_NAME
OUT_IMG_DIR = './out_images_BiGAN' #output image file
out_model_dir = './out_models_BiGAN' #output model file
CYCLE_LAMBDA = 1.0

try:
    os.mkdir('log')
    os.mkdir('out_graph')
    os.mkdir(OUT_IMG_DIR)
    os.mkdir(out_model_dir)
    os.mkdir('./out_images_Debug') #for debug
except:
    pass

make_datasets = Make_datasets(FILE_NAME, IMG_WIDTH, IMG_HEIGHT, SEED)
model = Model(NOISE_UNIT_NUM, IMG_CHANNEL, SEED, BASE_CHANNEL, KEEP_PROB_RATE)

z_ = tf.placeholder(tf.float32, [None, NOISE_UNIT_NUM], name='z_') #noise to generator
x_ = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL], name='x_') #image to classifier
d_dis_f_ = tf.placeholder(tf.float32, [None, 1], name='d_dis_g_') #target of discriminator related to generator
d_dis_r_ = tf.placeholder(tf.float32, [None, 1], name='d_dis_r_') #target of discriminator related to real image
is_training_ = tf.placeholder(tf.bool, name = 'is_training')

with tf.variable_scope('encoder_model'):
    z_enc = model.encoder(x_, reuse=False, is_training=is_training_)

with tf.variable_scope('decoder_model'):
    x_dec = model.decoder(z_, reuse=False, is_training=is_training_)
    x_z_x = model.decoder(z_enc, reuse=True, is_training=is_training_) # for cycle consistency

with tf.variable_scope('discriminator_model'):
    #stream around discriminator
    drop3_r, logits_r = model.discriminator(x_, z_enc, reuse=False, is_training=is_training_) #real pair
    drop3_f, logits_f = model.discriminator(x_dec, z_, reuse=True, is_training=is_training_) #real pair
    drop3_re, logits_re = model.discriminator(x_z_x, z_enc, reuse=True, is_training=is_training_) #fake pair

with tf.name_scope("loss"):
    loss_dis_f = tf.reduce_mean(tf.square(logits_f - d_dis_f_), name='Loss_dis_gen') #loss related to generator
    loss_dis_r = tf.reduce_mean(tf.square(logits_r - d_dis_r_), name='Loss_dis_rea') #loss related to real image

    #total loss
    loss_dis_total = loss_dis_f + loss_dis_r
    loss_dec_total = loss_dis_f
    loss_enc_total = loss_dis_r

with tf.name_scope("score"):
    l_g = tf.reduce_mean(tf.abs(x_ - x_z_x), axis=(1,2,3))
    l_FM = tf.reduce_mean(tf.abs(drop3_r - drop3_re), axis=1)
    score_A =  SCORE_ALPHA * l_g + (1.0 - SCORE_ALPHA) * l_FM

with tf.name_scope("optional_loss"):
    loss_dec_opt = loss_dec_total + CYCLE_LAMBDA * l_g
    loss_enc_opt = loss_enc_total + CYCLE_LAMBDA * l_g

tf.summary.scalar('loss_dis_total', loss_dis_total)
tf.summary.scalar('loss_dec_total', loss_dec_total)
tf.summary.scalar('loss_enc_total', loss_enc_total)
merged = tf.summary.merge_all()

# t_vars = tf.trainable_variables()
dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")
enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

with tf.name_scope("train"):
    train_dis = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.5).minimize(loss_dis_total, var_list=dis_vars
                                                                                , name='Adam_dis')
    train_dec = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.5).minimize(loss_dec_total, var_list=dec_vars
                                                                                , name='Adam_dec')
    train_enc = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.5).minimize(loss_enc_total, var_list=enc_vars
                                                                                , name='Adam_enc')
    train_dec_opt = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.5).minimize(loss_dec_opt, var_list=dec_vars
                                                                                , name='Adam_dec')
    train_enc_opt = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.5).minimize(loss_enc_opt, var_list=enc_vars
                                                                                , name='Adam_enc')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter(BOARD_DIR_NAME, sess.graph)

log_list = []
log_list.append(['epoch', 'AUC'])
#training loop
for epoch in range(0, EPOCH):
    sum_loss_dis_f = np.float32(0)
    sum_loss_dis_r = np.float32(0)
    sum_loss_dis_total = np.float32(0)
    sum_loss_dec_total = np.float32(0)
    sum_loss_enc_total = np.float32(0)

    len_data = make_datasets.make_data_for_1_epoch()

    for i in range(0, len_data, BATCH_SIZE):
        img_batch = make_datasets.get_data_for_1_batch(i, BATCH_SIZE)
        z = make_datasets.make_random_z_with_norm(NOISE_MEAN, NOISE_STDDEV, len(img_batch), NOISE_UNIT_NUM)
        tar_g_1 = make_datasets.make_target_1_0(1.0, len(img_batch)) #1 -> real
        tar_g_0 = make_datasets.make_target_1_0(0.0, len(img_batch)) #0 -> fake

        #train discriminator
        sess.run(train_dis, feed_dict={z_:z, x_: img_batch, d_dis_f_: tar_g_0, d_dis_r_: tar_g_1, is_training_:True})

        #train decoder
        sess.run(train_dec, feed_dict={z_:z, d_dis_f_: tar_g_1, is_training_:True})
        # sess.run(train_dec_opt, feed_dict={z_:z, x_: img_batch, d_dis_f_: tar_g_1, is_training_:True})


        #train encoder
        sess.run(train_enc, feed_dict={x_:img_batch, d_dis_r_: tar_g_0, is_training_:True})
        # sess.run(train_enc_opt, feed_dict={x_:img_batch, d_dis_r_: tar_g_0, is_training_:True})


        # loss for discriminator
        loss_dis_total_, loss_dis_r_, loss_dis_f_ = sess.run([loss_dis_total, loss_dis_r, loss_dis_f],
                                                             feed_dict={z_: z, x_: img_batch, d_dis_f_: tar_g_0,
                                                                        d_dis_r_: tar_g_1, is_training_:False})

        #loss for decoder
        loss_dec_total_ = sess.run(loss_dec_total, feed_dict={z_: z, d_dis_f_: tar_g_1, is_training_:False})

        #loss for encoder
        loss_enc_total_ = sess.run(loss_enc_total, feed_dict={x_: img_batch, d_dis_r_: tar_g_0, is_training_:False})

        #for tensorboard
        merged_ = sess.run(merged, feed_dict={z_:z, x_: img_batch, d_dis_f_: tar_g_0, d_dis_r_: tar_g_1, is_training_:False})

        summary_writer.add_summary(merged_, epoch)

        sum_loss_dis_f += loss_dis_f_
        sum_loss_dis_r += loss_dis_r_
        sum_loss_dis_total += loss_dis_total_
        sum_loss_dec_total += loss_dec_total_
        sum_loss_enc_total += loss_enc_total_

    print("----------------------------------------------------------------------")
    print("epoch = {:}, Encoder Total Loss = {:.4f}, Decoder Total Loss = {:.4f}, Discriminator Total Loss = {:.4f}".format(
        epoch, sum_loss_enc_total / len_data, sum_loss_dec_total / len_data, sum_loss_dis_total / len_data))
    print("Discriminator Real Loss = {:.4f}, Discriminator Generated Loss = {:.4f}".format(
        sum_loss_dis_r / len_data, sum_loss_dis_r / len_data))

    if epoch % VALID_SPAN == 0:
        # score_A_list = []
        score_A_np = np.zeros((0, 2), dtype=np.float32)
        val_data_num = len(make_datasets.valid_data_5_7)
        for i in range(0, val_data_num, BATCH_SIZE):
            img_batch, tars_batch = make_datasets.get_valid_data_for_1_batch(i, BATCH_SIZE)
            score_A_ = sess.run(score_A, feed_dict={x_:img_batch, is_training_:False})
            score_A_re = np.reshape(score_A_, (-1, 1))
            tars_batch_re = np.reshape(tars_batch, (-1, 1))

            score_A_np_tmp = np.concatenate((score_A_re, tars_batch_re), axis=1)
            score_A_np = np.concatenate((score_A_np, score_A_np_tmp), axis=0)

        tp, fp, tn, fn, precision, recall = Utility.compute_precision_recall(score_A_np)
        auc = Utility.make_ROC_graph(score_A_np, 'out_graph/' + LOGFILE_NAME, epoch)
        print("tp:{}, fp:{}, tn:{}, fn:{}, precision:{:.4f}, recall:{:.4f}, AUC:{:.4f}".format(tp, fp, tn, fn, precision, recall, auc))
        log_list.append([epoch, auc])

        img_batch_7, _ = make_datasets.get_valid_data_for_1_batch(0, 10)
        img_batch_5, _ = make_datasets.get_valid_data_for_1_batch(val_data_num - 11, 10)

        x_z_x_7 = sess.run(x_z_x, feed_dict={x_:img_batch_7, is_training_:False})
        x_z_x_5 = sess.run(x_z_x, feed_dict={x_:img_batch_5, is_training_:False})

        Utility.make_output_img(img_batch_5, img_batch_7, x_z_x_5, x_z_x_7, epoch, LOGFILE_NAME, OUT_IMG_DIR)
    #after learning
    Utility.save_list_to_csv(log_list, 'log/' + LOGFILE_NAME + '_auc.csv')

