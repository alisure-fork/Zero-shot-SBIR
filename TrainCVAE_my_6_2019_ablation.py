import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import layers
import keras.backend as k
from keras import callbacks
from keras.models import Model
from keras.optimizers import Adam
from alisuretool.Tools import Tools
import keras.backend.tensorflow_backend as ktf
from sklearn.neighbors import NearestNeighbors


class SketchyData(object):
    def __init__(self, ablation_5_1, n_x=4096, n_y=4096, n_z=1024, n_d=2048, test_split_ref=None):
        self.ablation_5_1 = ablation_5_1
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.n_d = n_d
        self.test_split_ref = test_split_ref
        pass

    @staticmethod
    def random_test_split_ref(data_path_alisure="./data/alisure", test_num=25):
        image_paths = list(np.load(os.path.join(data_path_alisure, 'image_paths.npy')))  # (12500, )
        _all_class = list(set([image_path.split('/')[-2] for image_path in image_paths]))
        test_split_ref = [_all_class[i] for i in np.random.permutation(len(_all_class))[0: test_num]]
        return test_split_ref

    # 读取所有的数据
    def _load_data(self, data_path_alisure="./data/alisure", data_path_ext="./data/ZSSBIR_data"):
        # image
        image_vgg_features = np.load(os.path.join(data_path_alisure, 'image_vgg_features.npy'))  # (12500, 4096)
        image_paths = list(np.load(os.path.join(data_path_alisure, 'image_paths.npy')))  # (12500, )
        # sketch
        sketch_vgg_features = np.load(os.path.join(data_path_alisure, 'sketch_vgg_features.npy'))  # (75479, 4096)
        sketch_model_features = np.load(os.path.join(data_path_alisure, 'sketch_model_features.npy'))  # (75479, 4096)
        sketch_paths = list(np.load(os.path.join(data_path_alisure, 'sketch_paths.npy')))  # (75479,)

        # extension
        image_ext_vgg_features = np.load(os.path.join(data_path_ext, 'image_ext_vgg_features.npy'))  # (73002, 4096)
        image_ext_paths = [_.decode() for _ in np.load(os.path.join(data_path_ext, 'image_ext_paths.npy'))]  # (73002, )

        if self.test_split_ref is None:  # 原始划分
            self.test_split_ref = [_.decode() for _ in np.load(os.path.join(data_path_ext, 'test_split_ref.npy'))]

        return (image_vgg_features, image_paths, sketch_vgg_features, sketch_model_features,
                sketch_paths, image_ext_vgg_features, image_ext_paths, self.test_split_ref)

    # 读取所有的数据
    @staticmethod
    def _load_data_old(data_path="./data/ZSSBIR_data"):
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

        print("train num is {}".format(len(train_sketch_paths)))
        print("test  num is {}".format(len(test_sketch_paths)))

        return train_sketch_paths, test_sketch_paths, train_classes, test_split_ref

    @staticmethod
    def get_image_path_by_sketch_path(sketch_path):
        temp_arr = sketch_path.replace('sketch', 'photo').split('-')
        _image_path = ''
        for _index in range(len(temp_arr) - 1):
            _image_path += temp_arr[_index] + '-'
        return _image_path[:-1] + '.jpg'

    # 划分特征
    def _split_data_features(self, sketch_paths, image_paths, image_ext_paths, train_sketch_paths, test_sketch_paths,
                             sketch_vgg_features, sketch_model_features, image_vgg_features,
                             image_ext_vgg_features, train_classes, n_x, n_y):

        # 1.
        # 形成一个反向索引：sketch
        sketch_path_index_tracker = {}
        for idx in range(len(sketch_paths)):
            sketch_path_index_tracker[sketch_paths[idx]] = idx
            pass

        is_two = True if self.ablation_5_1["vgg_sketch"] and self.ablation_5_1["model_sketch"] else False
        train_sketch_x = np.zeros((len(train_sketch_paths), n_y))
        for ii in range(len(train_sketch_paths)):
            index = sketch_path_index_tracker[train_sketch_paths[ii]]
            if is_two:
                train_sketch_x[ii, :] = np.append(sketch_vgg_features[index, :], sketch_model_features[index, :])
            else:
                if self.ablation_5_1["vgg_sketch"]:
                    train_sketch_x[ii, :] = sketch_vgg_features[index, :]
                elif self.ablation_5_1["model_sketch"]:
                    train_sketch_x[ii, :] = sketch_model_features[index, :]
                    pass
                pass
            pass

        test_sketch_x = np.zeros((len(test_sketch_paths), n_y))
        for ii in range(len(test_sketch_paths)):
            index = sketch_path_index_tracker[test_sketch_paths[ii]]
            if is_two:
                test_sketch_x[ii, :] = np.append(sketch_vgg_features[index, :], sketch_model_features[index, :])
            else:
                if self.ablation_5_1["vgg_sketch"]:
                    test_sketch_x[ii, :] = sketch_vgg_features[index, :]
                elif self.ablation_5_1["model_sketch"]:
                    test_sketch_x[ii, :] = sketch_model_features[index, :]
                    pass
                pass
            pass

        # 2.
        # 形成一个反向索引：image
        image_path_index_tracker = {}
        for idx in range(len(image_paths)):
            image_path_index_tracker[image_paths[idx]] = idx
            pass

        train_x_img = np.zeros((len(train_sketch_paths), n_x))
        for idx in range(len(train_sketch_paths)):
            image_path = self.get_image_path_by_sketch_path(train_sketch_paths[idx])
            image_idx = image_path_index_tracker[image_path]
            train_x_img[idx, :] = image_vgg_features[image_idx]
            pass

        test_image_path = []
        test_x_img = np.zeros((len(test_sketch_paths), n_x))
        for idx in range(len(test_sketch_paths)):
            image_path = self.get_image_path_by_sketch_path(test_sketch_paths[idx])
            image_idx = image_path_index_tracker[image_path]
            test_x_img[idx, :] = image_vgg_features[image_idx]
            test_image_path.append(image_path)
            pass

        # 3.
        # 形成一个反向索引：path ext
        test_image_paths_ext_index_tracker = {}
        for idx in range(len(image_ext_paths)):
            test_image_paths_ext_index_tracker[image_ext_paths[idx]] = idx
            pass

        # 不在训练集中的类别
        test_image_paths_ext = []
        for path in image_ext_paths:
            class_name = path.split('/')[-2]
            if class_name not in train_classes:
                test_image_paths_ext.append(path)
            pass

        test_img_vgg_features_ext = np.zeros((len(test_image_paths_ext), 4096))
        for idx in range(len(test_image_paths_ext)):
            original_index = test_image_paths_ext_index_tracker[test_image_paths_ext[idx]]
            test_img_vgg_features_ext[idx, :] = image_ext_vgg_features[original_index, :]
            pass

        # 用于可视化
        def save_data_for_vis():
            np.save("./data/test/ext_image.npy", test_img_vgg_features_ext)
            ext_label = [os.path.basename(os.path.split(test_image_path)[0]) for test_image_path in
                         test_image_paths_ext]
            np.save("./data/test/ext_image_class.npy", np.asarray(ext_label))

            np.save("./data/test/train_sketch.npy", train_sketch_x)
            np.save("./data/test/train_image.npy", train_x_img)
            train_label = [os.path.basename(os.path.split(train_sketch_path)[0]) for train_sketch_path in
                           train_sketch_paths]
            np.save("./data/test/train_image_class.npy", np.asarray(train_label))

            np.save("./data/test/test_image.npy", test_x_img)  # 目的是预测这个特征！如果直接用这个特征看准确率如何！这也许是准确率的上线！
            np.save("./data/test/test_sketch.npy", test_sketch_x)
            test_label = [os.path.basename(os.path.split(test_sketch_path)[0]) for test_sketch_path in
                          test_sketch_paths]
            np.save("./data/test/test_image_class.npy", np.asarray(test_label))
            pass

        # save_data_for_vis()

        return train_sketch_x, test_sketch_x, train_x_img, test_image_path, test_x_img, test_image_paths_ext, test_img_vgg_features_ext

    # 处理数据
    def get_data(self):
        # 加载数据：12500张图片和vgg特征，75479张素描图和vgg特征，增强数据集到73002张图片和vgg特征
        (image_vgg_features, image_paths, sketch_vgg_features, sketch_model_features,
         sketch_paths, image_ext_vgg_features, image_ext_paths, test_split_ref) = self._load_data()

        # 划分数据集：将原始素描图划分为训练集（62785）和测试集（75479 - 62785）
        train_sketch_paths, test_sketch_paths, train_classes, test_classes = self._split_data_path(sketch_paths,
                                                                                                   test_split_ref)

        # 划分特征：训练sketch，测试sketch，训练图片，测试图片，增强数据集中的测试类图片和vgg特征
        train_sketch_x, test_sketch_x, train_x_img, test_image_path, test_x_img, test_image_paths_ext, test_img_vgg_features_ext = \
            self._split_data_features(sketch_paths, image_paths, image_ext_paths, train_sketch_paths, test_sketch_paths,
                                      sketch_vgg_features, sketch_model_features, image_vgg_features,
                                      image_ext_vgg_features, train_classes, self.n_x, self.n_y)

        # 增强数据集中测试图片的类别
        test_image_ext_classes = np.array([path.split('/')[-2] for path in test_image_paths_ext])

        # 原始数据集中测试sketch的类别
        test_sketch_classes = np.array([path.split('/')[-2] for path in test_sketch_paths])

        # 原始数据集中训练sketch的类别
        train_sketch_classes = np.array([path.split('/')[-2] for path in train_sketch_paths])

        return (test_image_ext_classes, train_classes, train_sketch_classes, test_sketch_classes,
                test_img_vgg_features_ext, test_image_paths_ext,
                train_sketch_paths, train_sketch_x, train_x_img,
                test_sketch_paths, test_sketch_x, test_image_path, test_x_img)

    pass


class CVAE(object):

    def __init__(self, sketchy_data, ablation_5_2, max_epoch=25, batch_size=1024,
                 summary_path="summary", txt_path="log.txt"):
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.ablation_5_2 = ablation_5_2

        self.summary_path = summary_path
        self.txt_path = txt_path

        self.sketchy_data = sketchy_data
        self.n_x = sketchy_data.n_x
        self.n_y = sketchy_data.n_y
        self.n_d = sketchy_data.n_d
        self.n_z = sketchy_data.n_z

        self.hidden_size = 2048

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        ktf.set_session(session=self.sess)

        self.decoder, self.decoder_2, self.vae, self.vae_p2 = self.build()

        (self.image_classes, self.train_classes, self.train_sketch_classes, self.test_sketch_classes,
         self.img_vgg_features_ext, self.image_paths_ext,
         self.train_sketch_paths, self.train_sketch_x, self.train_x_img,
         self.test_sketch_paths, self.test_sketch_x, self.test_image_path, self.test_x_img) = sketchy_data.get_data()

        # Summary 1
        self.m_ap_op, self.precision_neigh_num_op, self.merged_summary_op = self._summary()
        self.summary_writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)
        pass

    @staticmethod
    def _summary():
        # Summary 2
        m_ap_op = tf.placeholder(dtype=tf.float32, shape=())
        precision_neigh_num_op = tf.placeholder(dtype=tf.float32, shape=())

        tf.summary.scalar("m_ap", m_ap_op)
        tf.summary.scalar("precision", precision_neigh_num_op)

        merged_summary_op = tf.summary.merge_all()
        return m_ap_op, precision_neigh_num_op, merged_summary_op

    def build(self):

        # 编码部分
        def net_encoder(_sketch_features, _image_features):
            input_combined = layers.concatenate([_image_features, _sketch_features])

            # Construct Encoder
            temp_h_q = layers.Dense(self.hidden_size * 2, activation='relu')(input_combined)
            temp_h_q_bn = layers.normalization.BatchNormalization()(temp_h_q)
            h_q_zd = layers.Dropout(rate=0.3)(temp_h_q_bn)
            h_q = layers.Dense(self.hidden_size, activation='relu')(h_q_zd)
            h_q_bn = layers.normalization.BatchNormalization()(h_q)

            # parameters of hidden variable
            _mu = layers.Dense(self.n_z, activation='tanh')(h_q_bn)
            _log_sigma = layers.Dense(self.n_z, activation='tanh')(h_q_bn)

            _encoder = Model(inputs=[_sketch_features, _image_features], outputs=[_mu])
            return _mu, _log_sigma, _encoder

        # 解码层
        decoder_1_hidden = layers.Dense(self.hidden_size, activation='relu')
        decoder_1_out = layers.Dense(self.n_x, activation='relu', name='decoder_out')

        # 解码部分：解码网P1
        def net_1_decoder(_features, _input_z):
            d_in = layers.concatenate([_input_z, _features])
            d_h = decoder_1_hidden(d_in)
            _d_out = decoder_1_out(d_h)
            return _d_out

        decoder_2_hidden = layers.Dense(self.hidden_size, activation='relu')
        decoder_2_out = layers.Dense(self.n_d, activation='relu', name='decoder_2_out')

        # 映射网P2
        def net_2_decoder(_features):
            d_h = decoder_2_hidden(_features)
            _d_out = decoder_2_out(d_h)
            return _d_out

        # 输入
        sketch_features = layers.Input(shape=[self.n_y], name='sketch_features')
        image_features = layers.Input(shape=[self.n_x], name='image_features')
        self.image_diff_features = layers.Input(shape=[self.n_x], name='image_diff_features')
        input_z = layers.Input(shape=[self.n_z], name='input_z')

        # 编码
        self.mu, self.log_sigma, encoder = net_encoder(sketch_features, image_features)

        # 测试时使用
        d_1_out = net_1_decoder(sketch_features, input_z)
        d_2_out = net_2_decoder(d_1_out)
        decoder = Model(inputs=[sketch_features, input_z], outputs=[d_1_out, d_2_out])
        d_2_out_test = net_2_decoder(image_features)
        decoder_2 = Model(inputs=[image_features], outputs=[d_2_out_test])

        # 训练时使用
        # 噪声
        sample_z = layers.Lambda(self._sample_z)([self.mu, self.log_sigma])
        # 重构Image
        d_out_recons = net_1_decoder(sketch_features, sample_z)  # P1
        self.d_2_out_recons = net_2_decoder(d_out_recons)  # P2
        self.d_2_out_same_recons = net_2_decoder(image_features)  # same
        self.d_2_out_diff_recons = net_2_decoder(self.image_diff_features)  # diff
        # 重构Sketch
        _recons_int = layers.Dense(self.hidden_size, activation='relu')(d_out_recons)
        recons_sketch = layers.Dense(self.n_y, activation='relu', name='recons_output')(_recons_int)

        # VAE模型
        vae = Model(inputs=[sketch_features, image_features, self.image_diff_features],
                    outputs=[d_out_recons, recons_sketch, self.d_2_out_recons])
        vae.compile(optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0),
                    loss={'decoder_out': self._d_triplet_loss,
                          'recons_output': self._vae_loss,
                          'decoder_2_out': self._d_2_triplet_loss},
                    loss_weights={'decoder_out': 100.0, 'recons_output': 1.0, 'decoder_2_out': 0.0})

        # VAE P2
        vae_p2 = Model(inputs=[sketch_features, image_features, self.image_diff_features],
                       outputs=[d_out_recons, recons_sketch, self.d_2_out_recons])

        # 微调或冻结
        # vae_p2.trainable = True
        # for layer in vae_p2.layers[:16]:
        #     layer.trainable = False

        vae_p2.compile(optimizer=Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0),
                       loss={'decoder_out': self._d_triplet_loss,
                             'recons_output': self._vae_loss,
                             'decoder_2_out': self._d_2_triplet_loss},
                       loss_weights={'decoder_out': 100.0, 'recons_output': 1.0, 'decoder_2_out': 100.0})

        return decoder, decoder_2, vae, vae_p2

    def _d_triplet_loss(self, y_true, y_pred):
        # P1
        d_1_loss = k.mean(k.square(y_pred - y_true), axis=1)
        if self.ablation_5_2["triplet_1"]:
            diff_recon = k.mean(k.square(y_pred - self.image_diff_features), axis=1)
            d_1_loss = k.maximum(2. + d_1_loss - diff_recon, 0.)
        return d_1_loss

    def _d_2_triplet_loss(self, y_true, y_pred):
        # P2
        d_2_loss = k.mean(k.square(y_pred - self.d_2_out_same_recons), axis=1)
        if self.ablation_5_2["triplet_2"]:
            diff_recon_2 = k.mean(k.square(y_pred - self.d_2_out_diff_recons), axis=1)
            d_2_loss = k.maximum(2. + d_2_loss - diff_recon_2, 0.)
        return d_2_loss

    def _vae_loss(self, y_true, y_pred):
        # sketch重构
        recon = k.mean(k.square(y_pred - y_true), axis=1)
        # vae
        kl = 0.5 * k.sum(k.exp(self.log_sigma) + k.square(self.mu) - 1. - self.log_sigma, axis=1)
        # sketch重构 + vae
        return recon + kl

    def _sample_z(self, args):
        mu, log_sigma = args
        eps = k.random_normal(shape=[self.n_z], mean=0., stddev=1.)
        return mu + k.exp(log_sigma / 2) * eps

    def test_old(self, epoch):

        def map_change(input_arr):
            dup = np.copy(input_arr)
            for _index in range(input_arr.shape[1]):
                if _index != 0:
                    dup[:, _index] = dup[:, _index - 1] + dup[:, _index]
            return np.multiply(dup, input_arr)

        # 噪声
        noise_z = np.random.normal(size=[len(self.test_sketch_paths), self.n_z])
        # 测试sketch
        sketch_features_test = np.asarray(
            [self.test_sketch_x[ii].copy() for ii in range(0, len(self.test_sketch_paths))])

        # 预测的图片的重构特征: noise + sketch -> image features
        predict_image_features, predict_image_features_2 = self.decoder.predict(
            {'sketch_features': sketch_features_test, 'input_z': noise_z}, verbose=2)

        # 200个最近邻
        neigh_num = 200
        # 增强数据集中测试图片的vgg特征
        nearest_neigh = NearestNeighbors(neigh_num, metric='cosine',
                                         algorithm='brute').fit(self.img_vgg_features_ext)

        # 求距离预测图片最近的测试图片：距离和索引
        distances, indices = nearest_neigh.kneighbors(predict_image_features)

        # 通过索引得到类别
        retrieved_classes = self.image_classes[indices]

        # 判断是否正确
        results = np.zeros(retrieved_classes.shape)
        for idx in range(results.shape[0]):
            results[idx] = (retrieved_classes[idx] == self.test_sketch_classes[idx])

        # 平均准确率
        precision_neigh_num = np.mean(np.mean(results, axis=1))

        # 计算mAP
        temp = [np.arange(neigh_num) for _ in range(retrieved_classes.shape[0])]
        m_ap_term = 1.0 / (np.stack(temp, axis=0) + 1)
        m_ap = np.mean(np.mean(np.multiply(map_change(results), m_ap_term), axis=1))

        # Summary 3
        summary_now = self.sess.run(self.merged_summary_op, feed_dict={
            self.m_ap_op: m_ap, self.precision_neigh_num_op: precision_neigh_num})
        self.summary_writer.add_summary(summary_now, global_step=epoch)

        # 输出
        Tools.print('', txt_path=self.txt_path)
        Tools.print('The mean precision@200 for test sketches is ' + str(precision_neigh_num), txt_path=self.txt_path)
        Tools.print('The mAP for test_sketches is ' + str(m_ap), txt_path=self.txt_path)

        return precision_neigh_num

    def test_simple(self, epoch):

        def map_change(input_arr):
            dup = np.copy(input_arr)
            for _index in range(input_arr.shape[1]):
                if _index != 0:
                    dup[:, _index] = dup[:, _index - 1] + dup[:, _index]
            return np.multiply(dup, input_arr)

        def cal_result(_predict_image_features, _img_vgg_features_ext, neigh_num=200):
            # 增强数据集中测试图片的vgg特征
            nearest_neigh = NearestNeighbors(neigh_num, metric='cosine', algorithm='brute').fit(_img_vgg_features_ext)

            # 求距离预测图片最近的测试图片：距离和索引
            _start_time = time.time()
            distances, indices = nearest_neigh.kneighbors(_predict_image_features)
            retrial_time = time.time() - _start_time

            # 通过索引得到类别
            retrieved_classes = self.image_classes[indices]

            # 判断是否正确
            results = np.zeros(retrieved_classes.shape)
            for idx in range(results.shape[0]):
                results[idx] = (retrieved_classes[idx] == self.test_sketch_classes[idx])

            # 平均准确率
            _precision_neigh_num = np.mean(np.mean(results, axis=1))

            # 计算mAP
            temp = [np.arange(neigh_num) for _ in range(retrieved_classes.shape[0])]
            m_ap_term = 1.0 / (np.stack(temp, axis=0) + 1)
            _m_ap = np.mean(np.mean(np.multiply(map_change(results), m_ap_term), axis=1))

            # Summary 3
            summary_now = self.sess.run(self.merged_summary_op, feed_dict={
                self.m_ap_op: _m_ap, self.precision_neigh_num_op: _precision_neigh_num})
            self.summary_writer.add_summary(summary_now, global_step=epoch)

            return _precision_neigh_num, _m_ap, retrial_time

        # 噪声
        noise_z = np.random.normal(size=[len(self.test_sketch_paths), self.n_z])
        # 测试sketch
        sketch_features_test = np.asarray(
            [self.test_sketch_x[ii].copy() for ii in range(0, len(self.test_sketch_paths))])

        # 预测的图片的重构特征: noise + sketch -> image features
        start_time = time.time()
        predict_image_features, predict_image_features_2 = self.decoder.predict(
            {'sketch_features': sketch_features_test, 'input_z': noise_z}, verbose=2)
        predict_time = time.time() - start_time
        predict_num = len(sketch_features_test)

        # P1空间计算
        Tools.print("", txt_path=self.txt_path)
        start_time = time.time()
        precision_neigh_num, m_ap, retrial_time = cal_result(predict_image_features, self.img_vgg_features_ext)
        Tools.print("{} {} {} {}".format("P1空间计算",
                                         precision_neigh_num, m_ap, time.time() - start_time), txt_path=self.txt_path)
        Tools.print("time predict {}, retrail is {}, all is {}, av is {}".format(
            predict_time, retrial_time, predict_time + retrial_time,
                                        (predict_time + retrial_time) / predict_num), txt_path=self.txt_path)

        # P2空间计算
        start_time = time.time()
        img_vgg_features_ext = self.decoder_2.predict({"image_features": self.img_vgg_features_ext}, verbose=2)
        precision_neigh_num, m_ap, retrial_time = cal_result(predict_image_features_2, img_vgg_features_ext)
        Tools.print("{} {} {} {}".format(
            "P2空间计算", precision_neigh_num, m_ap, time.time() - start_time), txt_path=self.txt_path)
        Tools.print("time predict is {}, retrail is {}, all is {}, av is {}".format(
            predict_time, retrial_time, predict_time + retrial_time,
                                        (predict_time + retrial_time) / predict_num), txt_path=self.txt_path)
        Tools.print("", txt_path=self.txt_path)
        pass

    def test(self, epoch):

        def map_change(input_arr):
            dup = np.copy(input_arr)
            for _index in range(input_arr.shape[1]):
                if _index != 0:
                    dup[:, _index] = dup[:, _index - 1] + dup[:, _index]
            return np.multiply(dup, input_arr)

        def cal_result(_predict_image_features, _img_vgg_features_ext, neigh_num=200):
            neigh_num = neigh_num if neigh_num < len(_img_vgg_features_ext) else len(_img_vgg_features_ext)
            # 增强数据集中测试图片的vgg特征
            nearest_neigh = NearestNeighbors(neigh_num, metric='cosine', algorithm='brute').fit(_img_vgg_features_ext)

            # 求距离预测图片最近的测试图片：距离和索引
            _start_time = time.time()
            distances, indices = nearest_neigh.kneighbors(_predict_image_features)
            retrial_time = time.time() - _start_time

            # 通过索引得到类别
            retrieved_classes = self.image_classes[indices]

            # 判断是否正确
            results = np.zeros(retrieved_classes.shape)
            for idx in range(results.shape[0]):
                results[idx] = (retrieved_classes[idx] == self.test_sketch_classes[idx])

            # 平均准确率
            _precision_neigh_num = np.mean(np.mean(results, axis=1))

            # 计算mAP
            temp = [np.arange(neigh_num) for _ in range(retrieved_classes.shape[0])]
            m_ap_term = 1.0 / (np.stack(temp, axis=0) + 1)
            _m_ap = np.mean(np.mean(np.multiply(map_change(results), m_ap_term), axis=1))

            # Summary 3
            summary_now = self.sess.run(self.merged_summary_op, feed_dict={
                self.m_ap_op: _m_ap, self.precision_neigh_num_op: _precision_neigh_num})
            self.summary_writer.add_summary(summary_now, global_step=epoch)

            return _precision_neigh_num, _m_ap, retrial_time

        # 噪声
        noise_z = np.random.normal(size=[len(self.test_sketch_paths), self.n_z])
        # 测试sketch
        sketch_features_test = np.asarray(
            [self.test_sketch_x[ii].copy() for ii in range(0, len(self.test_sketch_paths))])

        # 预测的图片的重构特征: noise + sketch -> image features
        start_time = time.time()
        predict_image_features, predict_image_features_2 = self.decoder.predict(
            {'sketch_features': sketch_features_test, 'input_z': noise_z}, verbose=2)
        predict_time = time.time() - start_time
        predict_num = len(sketch_features_test)

        for neigh_num in [100, 200]:
            # P1空间计算
            Tools.print("", self.txt_path)
            start_time = time.time()
            precision_neigh_num, m_ap, retrial_time = cal_result(predict_image_features,
                                                                 self.img_vgg_features_ext, neigh_num)
            Tools.print("{} {} {} {} {}".format(neigh_num, "P1空间计算",
                                                precision_neigh_num, m_ap, time.time() - start_time), self.txt_path)
            Tools.print("{} predict is {}, retrail is {}, all is {}, av is {}".format(
                neigh_num, predict_time, retrial_time,
                predict_time + retrial_time, (predict_time + retrial_time) / predict_num), self.txt_path)

            # P2空间计算
            start_time = time.time()
            img_vgg_features_ext = self.decoder_2.predict({"image_features": self.img_vgg_features_ext}, verbose=2)
            precision_neigh_num, m_ap, retrial_time = cal_result(predict_image_features_2,
                                                                 img_vgg_features_ext, neigh_num)
            Tools.print("{} {} {} {} {}".format(neigh_num, "P2空间计算",
                                                precision_neigh_num, m_ap, time.time() - start_time), self.txt_path)
            Tools.print("{} predict is {}, retrail is {}, all is {}, av is {}".format(
                neigh_num, predict_time, retrial_time,
                predict_time + retrial_time, (predict_time + retrial_time) / predict_num), self.txt_path)
            Tools.print("", self.txt_path)
        pass

    @staticmethod
    def my_generator(sketch_features, image_features, train_x_img, train_sketch_x, batch_size):
        _index = list(range(sketch_features.shape[0]))
        while True:
            np.random.shuffle(_index)
            for i in range(0, len(_index) - batch_size, batch_size):
                # 随机选择不同的图片特征
                random_index = _index[len(_index) - batch_size - i: len(_index) - i]

                x = {'sketch_features': sketch_features[_index[i: i + batch_size]],
                     'image_features': image_features[_index[i: i + batch_size]],
                     'image_diff_features': image_features[random_index]}

                y = [train_x_img[_index[i: i + batch_size]],
                     train_sketch_x[_index[i: i + batch_size]],
                     train_x_img[_index[i: i + batch_size]]]  # <=并没有使用

                yield x, y
            pass
        pass

    # 训练
    def run(self, model_1_file, model_2_file, load_weights=True, save_weights=True):

        def change_lr(lr, epoch):
            lr = lr * (1 / np.power(2, epoch / 4))
            Tools.print("now lr is {:8f}".format(lr), txt_path=self.txt_path)
            return lr

        if not os.path.exists(os.path.split(model_1_file)[0]):
            os.makedirs(os.path.split(model_1_file)[0])
        if not os.path.exists(os.path.split(model_2_file)[0]):
            os.makedirs(os.path.split(model_2_file)[0])

        # 载入模型
        if os.path.exists(model_1_file) and load_weights:
            self.vae.load_weights(model_1_file, skip_mismatch=True)

        tensor_board = callbacks.TensorBoard(log_dir=self.summary_path, histogram_freq=0, update_freq=1000)
        test_callback = callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self.test(epoch + 1))

        # 测试
        self.test(0)

        # 训练:vae
        learning_rate_scheduler_callback = callbacks.LearningRateScheduler(lambda epoch: change_lr(0.001, epoch))
        self.vae.fit_generator(generator=self.my_generator(
            self.train_sketch_x, self.train_x_img, self.train_x_img, self.train_sketch_x, self.batch_size),
            epochs=self.max_epoch, verbose=2, callbacks=[tensor_board, test_callback, learning_rate_scheduler_callback],
            steps_per_epoch=len(self.train_sketch_x) // self.batch_size)

        # 保存模型
        if save_weights:
            self.vae.save_weights(model_1_file)

        # 训练:vae_p2
        learning_rate_scheduler_callback = callbacks.LearningRateScheduler(lambda epoch: change_lr(0.0005, epoch))
        self.vae_p2.fit_generator(generator=self.my_generator(
            self.train_sketch_x, self.train_x_img, self.train_x_img, self.train_sketch_x, self.batch_size),
            epochs=self.max_epoch, verbose=2, callbacks=[tensor_board, test_callback, learning_rate_scheduler_callback],
            steps_per_epoch=len(self.train_sketch_x) // self.batch_size)

        # 测试
        self.test(self.max_epoch)

        # 保存模型
        if save_weights:
            self.vae.save_weights(model_2_file)

        pass

    # 测试
    def run2(self, model_1_file, model_2_file, load_weights=True):
        # 载入模型
        if os.path.exists(model_2_file) and load_weights:
            self.vae.load_weights(model_2_file, skip_mismatch=True)
        # 测试
        self.test2()
        pass

    def test2(self):

        def map_change(input_arr):
            dup = np.copy(input_arr)
            for _index in range(input_arr.shape[1]):
                if _index != 0:
                    dup[:, _index] = dup[:, _index - 1] + dup[:, _index]
            return np.multiply(dup, input_arr)

        def from_predict_test():
            # 噪声
            noise_z = np.random.normal(size=[len(self.test_sketch_paths), self.n_z])
            # 测试sketch
            sketch_features_test = np.asarray(
                [self.test_sketch_x[ii].copy() for ii in range(0, len(self.test_sketch_paths))])

            # 预测的图片的重构特征: noise + sketch -> image features
            _predict_image_features = self.decoder.predict(
                {'sketch_features': sketch_features_test, 'input_z': noise_z}, verbose=2)

            # 预测的特征可视化
            # np.save("./data/test/predict_image.npy", _predict_image_features)
            return _predict_image_features

        def from_predict_train():
            # 噪声
            noise_z = np.random.normal(size=[len(self.train_sketch_classes), self.n_z])
            # 训练sketch
            sketch_features_train = np.asarray(
                [self.train_sketch_x[ii].copy() for ii in range(0, len(self.train_sketch_classes))])

            # 预测的图片的重构特征: noise + sketch -> image features
            _predict_image_features = self.decoder.predict(
                {'sketch_features': sketch_features_train, 'input_z': noise_z}, verbose=2)

            # 预测的特征可视化
            # np.save("./data/test/predict_image.npy", _predict_image_features)
            return _predict_image_features

        def cal_result_1(_search_image_features, _search_image_class, _predict_image_features, _test_image_class):
            neigh_num = 200
            # 增强数据集中测试图片的vgg特征
            nearest_neigh = NearestNeighbors(neigh_num, metric='cosine',
                                             algorithm='brute').fit(_search_image_features)

            # 求距离预测图片最近的测试图片：距离和索引
            distances, indices = nearest_neigh.kneighbors(_predict_image_features)

            # 通过索引得到类别
            retrieved_classes = _search_image_class[indices]

            # 判断是否正确
            results = np.zeros(retrieved_classes.shape)
            for idx in range(results.shape[0]):
                results[idx] = (retrieved_classes[idx] == _test_image_class[idx])
                pass

            # 平均准确率
            _precision_neigh_num = np.mean(np.mean(results, axis=1))
            # 计算mAP
            temp = [np.arange(neigh_num) for _ in range(retrieved_classes.shape[0])]
            m_ap_term = 1.0 / (np.stack(temp, axis=0) + 1)
            _m_ap = np.mean(np.mean(np.multiply(map_change(results), m_ap_term), axis=1))

            return _precision_neigh_num, _m_ap

        def cal_result(_search_image_features, _search_image_class, _predict_image_features, _test_image_class, info):

            if len(_predict_image_features) == 2:
                _predict_image_features_1, _predict_image_features_2 = _predict_image_features
            else:
                _predict_image_features_1, _predict_image_features_2 = _predict_image_features, _predict_image_features

            # P1空间计算
            Tools.print(info, txt_path=self.txt_path)
            start_time = time.time()
            precision_neigh_num, m_ap = cal_result_1(_search_image_features, _search_image_class,
                                                     _predict_image_features_1, _test_image_class)
            Tools.print("{} {} {} {}".format(
                "P1空间计算", precision_neigh_num, m_ap, time.time() - start_time), txt_path=self.txt_path)

            # P2空间计算
            start_time = time.time()
            img_vgg_features_ext = self.decoder_2.predict({"image_features": _search_image_features}, verbose=2)
            precision_neigh_num, m_ap = cal_result_1(img_vgg_features_ext, _search_image_class,
                                                     _predict_image_features_2, _test_image_class)
            Tools.print("{} {} {} {}".format("P2空间计算", precision_neigh_num, m_ap,
                                             time.time() - start_time), txt_path=self.txt_path)
            Tools.print("", txt_path=self.txt_path)
            pass

        # 1. 预测测试图片的图像特征，在增强的数据集中检索  0.389,0.270
        predict_image_features = from_predict_test()
        test_image_class = self.test_sketch_classes
        search_image_features = self.img_vgg_features_ext
        search_image_class = self.image_classes
        cal_result(search_image_features, search_image_class, predict_image_features, test_image_class,
                   info="预测测试图片的图像特征，在增强的数据集中检索")

        # 2. 使用原始测试图像（从VGG中提取）的图像特征，在增强的数据集中检索  0.733,0.656
        predict_image_features = self.test_x_img
        test_image_class = self.test_sketch_classes
        search_image_features = self.img_vgg_features_ext
        search_image_class = self.image_classes
        cal_result(search_image_features, search_image_class, predict_image_features, test_image_class,
                   info="使用原始测试图像（从VGG中提取）的图像特征，在增强的数据集中检索")

        # 3. 预测测试图片的图像特征，在测试数据集中检索  0.445,0.340
        predict_image_features = from_predict_test()
        test_image_class = self.test_sketch_classes
        search_image_features = self.test_x_img
        search_image_class = self.test_sketch_classes
        cal_result(search_image_features, search_image_class, predict_image_features, test_image_class,
                   info="预测测试图片的图像特征，在测试数据集中检索")

        # 4. 使用原始测试图像（从VGG中提取）的图像特征，在测试数据集中检索  0.850,0.804
        predict_image_features = self.test_x_img
        test_image_class = self.test_sketch_classes
        search_image_features = self.test_x_img
        search_image_class = self.test_sketch_classes
        cal_result(search_image_features, search_image_class, predict_image_features, test_image_class,
                   info="使用原始测试图像（从VGG中提取）的图像特征，在测试数据集中检索")

        # 5. 预测训练图片的图像特征，在训练集中检索  0.924,0.905
        predict_image_features = from_predict_train()
        test_image_class = self.train_sketch_classes
        search_image_features = self.train_x_img
        search_image_class = self.train_sketch_classes
        cal_result(search_image_features, search_image_class, predict_image_features, test_image_class,
                   info="预测训练图片的图像特征，在训练集中检索")

        # 6. 使用原始训练图像（从VGG中提取）的图像特征，在训练集中检索  0.869,0.835
        predict_image_features = self.train_x_img
        test_image_class = self.train_sketch_classes
        search_image_features = self.train_x_img
        search_image_class = self.train_sketch_classes
        cal_result(search_image_features, search_image_class, predict_image_features, test_image_class,
                   info="使用原始训练图像（从VGG中提取）的图像特征，在训练集中检索")

        # 7. 预测测试图片的图像特征，在训练和测试集中检索  0.056,0.020
        predict_image_features = from_predict_test()
        test_image_class = self.test_sketch_classes
        search_image_features = np.append(self.test_x_img, self.train_x_img, axis=0)
        search_image_class = np.append(self.test_sketch_classes, self.train_sketch_classes, axis=0)
        cal_result(search_image_features, search_image_class, predict_image_features, test_image_class,
                   info="预测测试图片的图像特征，在训练和测试集中检索")

        # 8. 使用原始测试图像（从VGG中提取）的图像特征，在训练和测试集中检索  0.681,0.603
        predict_image_features = self.test_x_img
        test_image_class = self.test_sketch_classes
        search_image_features = np.append(self.test_x_img, self.train_x_img, axis=0)
        search_image_class = np.append(self.test_sketch_classes, self.train_sketch_classes, axis=0)
        cal_result(search_image_features, search_image_class, predict_image_features, test_image_class,
                   info="使用原始测试图像（从VGG中提取）的图像特征，在训练和测试集中检索")

        # 9. 使用原始训练图像（从VGG中提取）的图像特征，在训练和测试集中检索  0.851, 0.812
        predict_image_features = self.train_x_img
        test_image_class = self.train_sketch_classes
        search_image_features = np.append(self.test_x_img, self.train_x_img, axis=0)
        search_image_class = np.append(self.test_sketch_classes, self.train_sketch_classes, axis=0)
        cal_result(search_image_features, search_image_class, predict_image_features, test_image_class,
                   info="使用原始训练图像（从VGG中提取）的图像特征，在训练和测试集中检索")

        # 10. 预测训练图片的图像特征，在训练和测试集中检索  0.919,0.898
        predict_image_features = from_predict_train()
        test_image_class = self.train_sketch_classes
        search_image_features = np.append(self.test_x_img, self.train_x_img, axis=0)
        search_image_class = np.append(self.test_sketch_classes, self.train_sketch_classes, axis=0)
        cal_result(search_image_features, search_image_class, predict_image_features, test_image_class,
                   info="预测训练图片的图像特征，在训练和测试集中检索")

        pass

    # 查看检索结果
    def run3(self, model_1_file, model_2_file, load_weights=True):
        # 载入模型
        if os.path.exists(model_2_file) and load_weights:
            self.vae.load_weights(model_2_file, skip_mismatch=True)
        # 测试
        self.test3()
        pass

    def test3(self):

        def map_change(input_arr):
            dup = np.copy(input_arr)
            for _index in range(input_arr.shape[1]):
                if _index != 0:
                    dup[:, _index] = dup[:, _index - 1] + dup[:, _index]
            return np.multiply(dup, input_arr)

        def display_image(_test_image_paths, _test_sketch_paths, indices, result_path="./paper/show_2"):
            image_1_paths = []
            for index in range(0, len(_test_image_paths), 10):
                if index > 10000:
                    return
                now_sketch_path = _test_sketch_paths[index]
                now_image_path = _test_image_paths[index]
                if now_image_path in image_1_paths:
                    continue
                image_1_paths.append(now_image_path)

                now_path = os.path.join(result_path, "{}_{}".format(
                    index, os.path.splitext(os.path.basename(now_image_path))[0]))
                if not os.path.exists(now_path):
                    os.makedirs(now_path)
                    pass

                Image.open(now_image_path).save(os.path.join(now_path, "0.jpg"))
                Image.open(now_sketch_path).save(os.path.join(now_path, "0_sketch.jpg"))

                image_2_paths = []
                for sketch_index_index, sketch_index in enumerate(indices[index]):
                    now_sketch_path = _test_sketch_paths[sketch_index]
                    now_image_path = _test_image_paths[sketch_index]
                    if now_image_path in image_2_paths:
                        continue
                    image_2_paths.append(now_image_path)
                    Image.open(now_sketch_path).save(os.path.join(now_path, "{}_{}".format(
                        sketch_index_index, os.path.basename(now_sketch_path))))
                    Image.open(now_image_path).save(os.path.join(now_path, "{}_{}".format(
                        sketch_index_index, os.path.basename(now_image_path))))
                    pass
                pass

            pass

        def cal_result_1(_test_image_paths, _search_image_features, _test_sketch_paths, _predict_image_features,
                         _search_image_class, _test_image_class, is_display=False):
            neigh_num = 200
            # 增强数据集中测试图片的vgg特征
            nearest_neigh = NearestNeighbors(neigh_num, metric='cosine',
                                             algorithm='brute').fit(_search_image_features)

            # 求距离预测图片最近的测试图片：距离和索引
            distances, indices = nearest_neigh.kneighbors(_predict_image_features)

            # 显示检索到的图像
            if is_display:
                display_image(_test_image_paths, _test_sketch_paths, indices)

            # 通过索引得到类别
            retrieved_classes = _search_image_class[indices]

            # 判断是否正确
            results = np.zeros(retrieved_classes.shape)
            for idx in range(results.shape[0]):
                results[idx] = (retrieved_classes[idx] == _test_image_class[idx])
                pass

            # 平均准确率
            _precision_neigh_num = np.mean(np.mean(results, axis=1))
            # 计算mAP
            temp = [np.arange(neigh_num) for _ in range(retrieved_classes.shape[0])]
            m_ap_term = 1.0 / (np.stack(temp, axis=0) + 1)
            _m_ap = np.mean(np.mean(np.multiply(map_change(results), m_ap_term), axis=1))

            return _precision_neigh_num, _m_ap

        def cal_result(_test_image_paths, _search_image_features, _test_sketch_paths, _predict_image_features,
                       _search_image_class, _test_image_class):

            if len(_predict_image_features) == 2:
                _predict_image_features_1, _predict_image_features_2 = _predict_image_features
            else:
                _predict_image_features_1, _predict_image_features_2 = _predict_image_features, _predict_image_features

            # P1空间计算
            start_time = time.time()
            precision_neigh_num, m_ap = cal_result_1(_test_image_paths, _search_image_features,
                                                     _test_sketch_paths, _predict_image_features_1,
                                                     _search_image_class, _test_image_class, is_display=False)
            Tools.print("{} {} {} {}".format(
                "P1空间计算", precision_neigh_num, m_ap, time.time() - start_time), txt_path=self.txt_path)

            # P2空间计算
            start_time = time.time()
            img_vgg_features_ext = self.decoder_2.predict({"image_features": _search_image_features}, verbose=2)
            precision_neigh_num, m_ap = cal_result_1(_test_image_paths, img_vgg_features_ext,
                                                     _test_sketch_paths, _predict_image_features_2,
                                                     _search_image_class, _test_image_class)
            Tools.print("{} {} {} {}".format(
                "P2空间计算", precision_neigh_num, m_ap, time.time() - start_time), txt_path=self.txt_path)
            Tools.print("", txt_path=self.txt_path)
            pass

        # 1. 预测测试图片的图像特征，在增强的数据集中检索  0.389,0.270
        test_image_class = self.test_sketch_classes
        search_image_class = self.test_sketch_classes

        test_image_paths = self.test_image_path  # 检索数据集图像地址
        search_image_features = self.test_x_img  # 检索数据集特征

        test_sketch_paths = self.test_sketch_paths  # 待检索素描图地址
        test_sketch_x = self.test_sketch_x  # 待检索图像地址

        predict_image_features = self.decoder.predict({
            'sketch_features': np.asarray([test_sketch_x[ii].copy() for ii in range(0, len(test_sketch_paths))]),
            'input_z': np.random.normal(size=[len(test_sketch_paths), self.n_z])}, verbose=2)  # 待检索图像

        cal_result(test_image_paths, search_image_features, test_sketch_paths, predict_image_features,
                   search_image_class, test_image_class)
        pass

    pass


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ablation_5_1 = {"vgg_sketch": True, "model_sketch": True}
    ablation_5_2 = {"triplet_1": True, "triplet_2": True}
    # ablation_n_d = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    ablation_n_d = [1024]
    run_name = "triplet_6_2019"
    batch_size = 1024

    for random_index in range(5):
        # 随机划分类别
        test_split_ref = SketchyData.random_test_split_ref(test_num=25)
        model_result_path = "./model/model_random_split5"

        for n_d in ablation_n_d:
            Tools.print("")
            name = "{}_{}_{}_{}_{}_{}_{}_{}".format(run_name, random_index, ablation_5_1["vgg_sketch"],
                                                    ablation_5_1["model_sketch"], ablation_5_2["triplet_1"],
                                                    ablation_5_2["triplet_2"], n_d, batch_size)
            now_path = Tools.new_dir("{}/{}".format(model_result_path, name))
            txt_path = "{}/log.txt".format(now_path)

            Tools.print("name = {}".format(name), txt_path=txt_path)
            Tools.print("test_split_ref = {}".format(test_split_ref), txt_path=txt_path)
            Tools.print("ablation_5_1 = [vgg_sketch: {}, model_sketch: {}]".format(
                ablation_5_1["vgg_sketch"], ablation_5_1["model_sketch"]), txt_path=txt_path)
            Tools.print("ablation_5_2 = [triplet_1: {}, triplet_2: {}]".format(
                ablation_5_2["triplet_1"], ablation_5_2["triplet_2"]), txt_path=txt_path)

            tf.reset_default_graph()
            n_y = 4096 * 2 if ablation_5_1["vgg_sketch"] and ablation_5_1["model_sketch"] else 4096
            sketchy_data = SketchyData(ablation_5_1, n_x=4096, n_y=n_y, n_z=1024, n_d=n_d,
                                       test_split_ref=test_split_ref)

            cvae = CVAE(sketchy_data=sketchy_data, ablation_5_2=ablation_5_2, max_epoch=20,
                        batch_size=batch_size, summary_path="{}/summary".format(now_path), txt_path=txt_path)

            cvae.run(model_1_file="{}/model_1.h5".format(now_path),
                     model_2_file="{}/model_2.h5".format(now_path), load_weights=False)

            # 保存检索的结果
            # cvae.run3(model_1_file="model/{}/model_1.h5".format(name),
            #           model_2_file="model/{}/model_2.h5".format(name), load_weights=True)
            pass
        pass

    pass


if __name__ == '__main__':
    main()

    pass
