import numpy as np
import os
import random

class Make_datasets_MNIST():

    def __init__(self, file_name, img_width, img_height, seed):
        self.filename = file_name
        self.img_width = img_width
        self.img_height = img_height
        self.seed = seed
        x_train, x_test, x_valid, y_train, y_test, y_valid = self.read_MNIST_npy(self.filename)
        self.train_np = np.concatenate((y_train.reshape(-1,1), x_train), axis=1).astype(np.float32)
        self.test_np = np.concatenate((y_test.reshape(-1,1), x_test), axis=1).astype(np.float32)
        self.valid_np = np.concatenate((y_valid.reshape(-1,1), x_valid), axis=1).astype(np.float32)
        print("self.train_np.shape, ", self.train_np.shape)
        print("self.test_np.shape, ", self.test_np.shape)
        print("self.valid_np.shape, ", self.valid_np.shape)
        print("np.max(x_train), ", np.max(x_train))
        print("np.min(x_train), ", np.min(x_train))
        self.train_data_5, self.train_data_7 = self.divide_MNIST_by_digit(self.train_np, 5, 7)
        print("self.train_data_5.shape, ", self.train_data_5.shape)
        print("self.train_data_7.shape, ", self.train_data_7.shape)
        self.valid_data_5, self.valid_data_7 = self.divide_MNIST_by_digit(self.valid_np, 5, 7)
        print("self.valid_data_5.shape, ", self.valid_data_5.shape)
        print("self.valid_data_7.shape, ", self.valid_data_7.shape)
        self.valid_data_5_7 = np.concatenate((self.train_data_7, self.valid_data_7, self.valid_data_5))

        random.seed(self.seed)
        np.random.seed(self.seed)


    def read_MNIST_npy(self, filename):
        mnist_npz = np.load(filename)
        print("type(mnist_npz), ", type(mnist_npz))
        print("mnist_npz.keys(), ", mnist_npz.keys())
        print("mnist_npz['x_train'].shape, ", mnist_npz['x_train'].shape)
        print("mnist_npz['x_test'].shape, ", mnist_npz['x_test'].shape)
        print("mnist_npz['x_valid'].shape, ", mnist_npz['x_valid'].shape)
        print("mnist_npz['y_train'].shape, ", mnist_npz['y_train'].shape)
        print("mnist_npz['y_test'].shape, ", mnist_npz['y_test'].shape)
        print("mnist_npz['y_valid'].shape, ", mnist_npz['y_valid'].shape)
        x_train = mnist_npz['x_train']
        x_test = mnist_npz['x_test']
        x_valid = mnist_npz['x_valid']
        y_train = mnist_npz['y_train']
        y_test = mnist_npz['y_test']
        y_valid = mnist_npz['y_valid']
        return x_train, x_test, x_valid, y_train, y_test, y_valid


    def get_file_names(self, dir_name):
        target_files = []
        for root, dirs, files in os.walk(dir_name):
            targets = [os.path.join(root, f) for f in files]
            target_files.extend(targets)

        return target_files

    def divide_MNIST_by_digit(self, train_np, data1_num, data2_num):
        data_1 = train_np[train_np[:,0] == data1_num]
        data_2 = train_np[train_np[:,0] == data2_num]

        return data_1, data_2



    def read_data(self, d_y_np, width, height):
        tars = []
        images = []
        for num, d_y_1 in enumerate(d_y_np):
            image = d_y_1[1:].reshape(width, height, 1)
            tar = d_y_1[0]
            images.append(image)
            tars.append(tar)

        return np.asarray(images), np.asarray(tars)


    def normalize_data(self, data):
        # data0_2 = data / 127.5
        # data_norm = data0_2 - 1.0
        data_norm = (data * 2.0) - 1.0 #applied for tanh

        return data_norm


    def make_data_for_1_epoch(self):
        self.filename_1_epoch = np.random.permutation(self.train_data_5)

        return len(self.filename_1_epoch)


    def get_data_for_1_batch(self, i, batchsize):
        filename_batch = self.filename_1_epoch[i:i + batchsize]
        images, _ = self.read_data(filename_batch, self.img_width, self.img_height)
        images_n = self.normalize_data(images)
        return images_n

    def get_valid_data_for_1_batch(self, i, batchsize):
        filename_batch = self.valid_data_5_7[i:i + batchsize]
        images, tars = self.read_data(filename_batch, self.img_width, self.img_height)
        images_n = self.normalize_data(images)
        return images_n, tars

    def make_random_z_with_norm(self, mean, stddev, data_num, unit_num):
        norms = np.random.normal(mean, stddev, (data_num, unit_num))
        # tars = np.zeros((data_num, 1), dtype=np.float32)
        return norms


    def make_target_1_0(self, value, data_num):
        if value == 0.0:
            target = np.zeros((data_num, 1), dtype=np.float32)
        elif value == 1.0:
            target = np.ones((data_num, 1), dtype=np.float32)
        else:
            print("target value error")
        return target

def check_mnist_npz(filename):
    mnist_npz = np.load(filename)
    print("type(mnist_npz), ", type(mnist_npz))
    print("mnist_npz.keys(), ", mnist_npz.keys())
    print("mnist_npz['x_train'].shape, ", mnist_npz['x_train'].shape)
    print("mnist_npz['x_test'].shape, ", mnist_npz['x_test'].shape)
    print("mnist_npz['x_valid'].shape, ", mnist_npz['x_valid'].shape)
    print("mnist_npz['y_train'].shape, ", mnist_npz['y_train'].shape)
    print("mnist_npz['y_test'].shape, ", mnist_npz['y_test'].shape)
    print("mnist_npz['y_valid'].shape, ", mnist_npz['y_valid'].shape)


if __name__ == '__main__':
    #debug
    FILE_NAME = './mnist.npz'
    # check_mnist_npz(FILE_NAME)
    # make_datasets = Make_datasets_MNIST(FILE_NAME, 28, 28, 1234, 0.0, 0.2, 6000, 512)
