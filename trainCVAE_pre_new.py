from keras.layers import Input, Dense, Lambda, Dropout, concatenate
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import keras.backend as K
import tensorflow as tf
import os
import keras.backend.tensorflow_backend as KTF
from sklearn.neighbors import NearestNeighbors
from keras.layers.normalization import BatchNormalization
import pandas as pd


class SketchyData(object):

    def __init__(self, n_x=4096, n_y=4096):
        self.n_x = n_x
        self.n_y = n_y
        pass

    # 读取所有的数据
    @staticmethod
    def _load_data(data_path="./data/ZSSBIR_data"):
        # first load the image features and imagePaths
        image_vgg_features = np.load(os.path.join(data_path, 'image_vgg_features.npy'))  # (12500, 4096)
        image_paths = [_.decode() for _ in np.load(os.path.join(data_path, 'image_paths.npy'))]  # (12500, )

        # next load the sketch_paths
        sketch_vgg_features = np.load(os.path.join(data_path, 'sketch_vgg_features.npy'))  # (75479, 4096)
        sketch_paths = [_.decode() for _ in np.load(os.path.join(data_path, 'sketch_paths.npy'))]  # (75479,)

        # next load the image extension dataset
        image_ext_vgg_features = np.load(os.path.join(data_path, 'image_ext_vgg_features.npy'))  # (73002, 4096)
        image_ext_paths = [_.decode() for _ in np.load(os.path.join(data_path, 'image_ext_paths.npy'))]  # (73002, )

        test_split_ref = [_.decode() for _ in np.load(os.path.join(data_path, 'test_split_ref.npy'))]

        return (image_vgg_features, image_paths, sketch_vgg_features, sketch_paths,
                image_ext_vgg_features, image_ext_paths, test_split_ref)

    # 划分数据
    @staticmethod
    def _split_data_path(sketch_paths, test_split_ref):
        # 按照类别划分
        sketch_paths_per_class = {}
        for sketch_path in sketch_paths:
            class_name = sketch_path.split('/')[-2]
            if class_name not in sketch_paths_per_class:
                sketch_paths_per_class[class_name] = []
            sketch_paths_per_class[class_name].append(sketch_path)
            pass

        # 训练集
        train_sketch_paths = sketch_paths.copy()
        # 测试集
        test_sketch_paths = np.array([])

        # 划分
        train_classes = []
        for class_name in sketch_paths_per_class:
            if class_name not in test_split_ref:
                train_classes.append(class_name)
            else:
                test_sketch_paths = np.append(test_sketch_paths, sketch_paths_per_class[class_name])
                for test_path in sketch_paths_per_class[class_name]:
                    train_sketch_paths.remove(test_path)
            pass

        print(len(train_sketch_paths))
        print(len(test_sketch_paths))

        test_sketch_paths = test_sketch_paths[0: 5000]
        return train_sketch_paths, test_sketch_paths, train_classes, test_split_ref

    # 划分特征
    @staticmethod
    def _split_data_features(sketch_paths, image_paths, image_ext_paths, train_sketch_paths, test_sketch_paths,
                             sketch_vgg_features, image_vgg_features, image_ext_vgg_features, train_classes, n_x, n_y):

        def get_image_path(sketch_path):
            temp_arr = sketch_path.replace('sketch', 'photo').split('-')
            image_path = ''
            for idx in range(len(temp_arr) - 1):
                image_path += temp_arr[idx] + '-'
            return image_path[:-1] + '.jpg'

        # 1.
        # 形成一个反向索引：sketch
        sketch_path_index_tracker = {}
        for idx in range(len(sketch_paths)):
            sketch_path_index_tracker[sketch_paths[idx]] = idx

        train_sketch_x = np.zeros((len(train_sketch_paths), n_y))
        for ii in range(len(train_sketch_paths)):
            index = sketch_path_index_tracker[train_sketch_paths[ii]]
            train_sketch_x[ii, :] = sketch_vgg_features[index, :]

        test_sketch_x = np.zeros((len(test_sketch_paths), n_y))
        for ii in range(len(test_sketch_paths)):
            index = sketch_path_index_tracker[test_sketch_paths[ii]]
            test_sketch_x[ii, :] = sketch_vgg_features[index, :]

        # 2.
        # 形成一个反向索引：image
        image_path_index_tracker = {}
        for idx in range(len(image_paths)):
            image_path_index_tracker[image_paths[idx]] = idx

        train_x_img = np.zeros((len(train_sketch_paths), n_x))
        for idx in range(len(train_sketch_paths)):
            image_path = get_image_path(train_sketch_paths[idx])
            image_idx = image_path_index_tracker[image_path]
            train_x_img[idx, :] = image_vgg_features[image_idx]

        test_x_img = np.zeros((len(test_sketch_paths), n_x))
        for idx in range(len(test_sketch_paths)):
            image_path = get_image_path(test_sketch_paths[idx])
            image_idx = image_path_index_tracker[image_path]
            test_x_img[idx, :] = image_vgg_features[image_idx]

        # 3.
        # 形成一个反向索引：path ext
        image_paths_ext_index_tracker = {}
        for idx in range(len(image_ext_paths)):
            image_paths_ext_index_tracker[image_ext_paths[idx]] = idx

        # 不在训练集中的类别
        image_paths_ext = []
        for path in image_ext_paths:
            class_name = path.split('/')[-2]
            if class_name not in train_classes:
                image_paths_ext.append(path)
            pass

        img_vgg_features_ext = np.zeros((len(image_paths_ext), 4096))
        for idx in range(len(image_paths_ext)):
            original_index = image_paths_ext_index_tracker[image_paths_ext[idx]]
            img_vgg_features_ext[idx, :] = image_ext_vgg_features[original_index, :]

        return train_sketch_x, test_sketch_x, train_x_img, test_x_img, image_paths_ext, img_vgg_features_ext

    # 处理数据
    def get_data(self):
        # 加载数据
        (image_vgg_features, image_paths, sketch_vgg_features, sketch_paths,
         image_ext_vgg_features, image_ext_paths, test_split_ref) = self._load_data()

        # 划分数据集
        train_sketch_paths, test_sketch_paths, train_classes, test_classes = self._split_data_path(sketch_paths,
                                                                                                   test_split_ref)

        # 划分特征
        train_sketch_x, test_sketch_x, train_x_img, test_x_img, image_paths_ext, img_vgg_features_ext = \
            self._split_data_features(sketch_paths, image_paths, image_ext_paths, train_sketch_paths, test_sketch_paths,
                                      sketch_vgg_features, image_vgg_features, image_ext_vgg_features, train_classes,
                                      self.n_x, self.n_y)

        image_ext_classes = np.array([path.split('/')[-2] for path in image_paths_ext])
        test_sketch_classes = np.array([path.split('/')[-2] for path in test_sketch_paths])

        return (image_ext_classes, train_classes, test_sketch_classes, img_vgg_features_ext, image_paths_ext,
                train_sketch_x, train_x_img, train_sketch_x, test_sketch_paths, test_sketch_x)

    pass


class CVAE(object):

    def __init__(self, sketchy_data):

        self.max_epoch = 25
        self.batch_size = 512

        self.n_x = 4096
        self.n_y = 4096
        self.n_z = 1024

        self.internal_size = 2048

        # Build a nearest neighbour classifier
        self.neigh_num = 200

        self.prec = 0

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))

        KTF.set_session(session=self.sess)

        self.decoder, self.vae = self.build()

        (self.image_classes, self.train_classes, self.test_sketch_classes, self.img_vgg_features_ext,
         self.image_paths_ext, self.train_sketch_x, self.train_x_img, self.train_sketch_x,
         self.test_sketch_paths, self.test_sketch_x) = sketchy_data.get_data()
        pass

    def build(self):
        sketch_features = Input(shape=[self.n_y], name='sketch_features')
        image_features = Input(shape=[self.n_x], name='image_features')
        input_combined = concatenate([image_features, sketch_features])

        # Construct Encoder
        temp_h_q = Dense(self.internal_size * 2, activation='relu')(input_combined)
        temp_h_q_bn = BatchNormalization()(temp_h_q)
        h_q_zd = Dropout(rate=0.3)(temp_h_q_bn)
        h_q = Dense(self.internal_size, activation='relu')(h_q_zd)
        h_q_bn = BatchNormalization()(h_q)

        # parameters of hidden variable
        self.mu = Dense(self.n_z, activation='tanh')(h_q_bn)
        self.log_sigma = Dense(self.n_z, activation='tanh')(h_q_bn)

        # concatenate sampled z and conditional input i.e. sketch
        z = Lambda(self._sample_z)([self.mu, self.log_sigma])
        z_cond = concatenate([z, sketch_features])

        # Define layers
        decoder_hidden = Dense(self.internal_size, activation='relu')
        decoder_out = Dense(self.n_x, activation='relu', name='decoder_out')

        # construct Decoder
        h_p = decoder_hidden(z_cond)
        reconstr = decoder_out(h_p)

        # Form models
        encoder = Model(inputs=[sketch_features, image_features], outputs=[self.mu])

        input_z = Input(shape=[self.n_z], name='input_z')
        d_in = concatenate([input_z, sketch_features])
        d_h = decoder_hidden(d_in)
        d_out = decoder_out(d_h)
        decoder = Model(inputs=[sketch_features, input_z], outputs=[d_out])

        # Predict the attribute again to enforce its usage
        attr_int = Dense(self.internal_size, activation='relu')(reconstr)
        attr_recons = Dense(self.n_y, activation='relu', name='recons_output')(attr_int)

        # Form the VAE model
        vae = Model(inputs=[sketch_features, image_features], outputs=[reconstr, attr_recons])

        # 打印网络summary
        encoder.summary()
        decoder.summary()

        # 配置训练模型
        vae.compile(optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0),
                    loss={'decoder_out': self._vae_loss, 'recons_output': 'mean_squared_error'},
                    loss_weights={'decoder_out': 1.0, 'recons_output': 10.0})

        return decoder, vae

    def _sample_z(self, args):
        mu, log_sigma = args
        eps = K.random_normal(shape=[self.n_z], mean=0., stddev=1.)
        return mu + K.exp(log_sigma / 2) * eps

    def _vae_loss(self, y_true, y_pred):
        # E[log P(X|z)]
        recon = K.mean(K.square(y_pred - y_true), axis=1)
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl = 0.5 * K.sum(K.exp(self.log_sigma) + K.square(self.mu) - 1. - self.log_sigma, axis=1)
        return recon + kl

    def find_precision(self):

        def map_change(input_arr):
            dup = np.copy(input_arr)
            for idx in range(input_arr.shape[1]):
                if (idx != 0):
                    dup[:, idx] = dup[:, idx - 1] + dup[:, idx]
            return np.multiply(dup, input_arr)

        # z + sketch 100份
        random_z_per_sketch = 100
        noise_ip = np.random.normal(size=[random_z_per_sketch * len(self.test_sketch_paths), self.n_z])
        sketch_ip = np.zeros([random_z_per_sketch * len(self.test_sketch_paths), self.n_y])
        for ii in range(0, len(self.test_sketch_paths)):
            for jj in range(0, random_z_per_sketch):
                sketch_ip[ii * random_z_per_sketch + jj] = self.test_sketch_x[ii]
            pass

        print('Predicting...')
        pred_image_features = self.decoder.predict({'sketch_features': sketch_ip, 'input_z': noise_ip}, verbose=1)
        # 对预测求平均
        avg_pred_img_features = np.zeros([len(self.test_sketch_paths), self.n_x])
        for ii in range(0, len(self.test_sketch_paths)):
            avg_pred_img_features[ii] = np.mean(pred_image_features[
                                             ii * random_z_per_sketch:(ii + 1) * random_z_per_sketch], axis=0)
            pass

        # 200个最近邻
        nbrs = NearestNeighbors(self.neigh_num, metric='cosine', algorithm='brute').fit(self.img_vgg_features_ext)
        distances, indices = nbrs.kneighbors(avg_pred_img_features)
        retrieved_classes = self.image_classes[indices]

        results = np.zeros(retrieved_classes.shape)
        for idx in range(results.shape[0]):
            results[idx] = (retrieved_classes[idx] == self.test_sketch_classes[idx])

        precision_200 = np.mean(results, axis=1)
        temp = [np.arange(200) for _ in range(results.shape[0])]
        m_ap_term = 1.0 / (np.stack(temp, axis=0) + 1)
        m_ap = np.mean(np.multiply(map_change(results), m_ap_term), axis=1)

        print('')
        print('The mean precision@200 for test sketches is ' + str(np.mean(precision_200)))
        print('The mAP for test_sketches is ' + str(np.mean(m_ap)))

        return np.mean(precision_200)

    def run(self):
        # 训练
        self.vae.fit({'sketch_features': self.train_sketch_x,
                      'image_features': self.train_x_img},
                     [self.train_x_img, self.train_sketch_x], batch_size=self.batch_size, nb_epoch=self.max_epoch)

        # 预测
        self.find_precision()
        pass

    pass


if __name__ == '__main__':
    CVAE(sketchy_data=SketchyData()).run()
