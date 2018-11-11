import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras import callbacks
from keras.models import Model
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
from sklearn.neighbors import NearestNeighbors
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Lambda, Dropout, concatenate


class SketchyData(object):

    def __init__(self, n_x=4096, n_y=4096):
        self.n_x = n_x
        self.n_y = n_y
        pass

    # 读取所有的数据
    @staticmethod
    def _load_data(data_path_alisure="./data/alisure", data_path_ext="./data/ZSSBIR_data"):
        # first load the image features and imagePaths
        image_vgg_features = np.load(os.path.join(data_path_alisure, 'image_vgg_features.npy'))  # (12500, 4096)
        image_paths = list(np.load(os.path.join(data_path_alisure, 'image_paths.npy')))  # (12500, )

        # next load the sketch_paths
        # sketch_features = np.load(os.path.join(data_path_alisure, 'sketch_vgg_features.npy'))  # (75479, 4096)
        sketch_features = np.load(os.path.join(data_path_alisure, 'sketch_model_features2.npy'))  # (75479, 4096)
        sketch_paths = list(np.load(os.path.join(data_path_alisure, 'sketch_paths.npy')))  # (75479,)

        # next load the image extension dataset
        image_ext_vgg_features = np.load(os.path.join(data_path_ext, 'image_ext_vgg_features.npy'))  # (73002, 4096)
        image_ext_paths = [_.decode() for _ in np.load(os.path.join(data_path_ext, 'image_ext_paths.npy'))]  # (73002, )

        test_split_ref = [_.decode() for _ in np.load(os.path.join(data_path_ext, 'test_split_ref.npy'))]

        return (image_vgg_features, image_paths, sketch_features, sketch_paths,
                image_ext_vgg_features, image_ext_paths, test_split_ref)

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

    # 划分特征
    @staticmethod
    def _split_data_features(sketch_paths, image_paths, image_ext_paths, train_sketch_paths, test_sketch_paths,
                             sketch_features, image_vgg_features, image_ext_vgg_features, train_classes, n_x, n_y):

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
            train_sketch_x[ii, :] = sketch_features[index, :]

        test_sketch_x = np.zeros((len(test_sketch_paths), n_y))
        for ii in range(len(test_sketch_paths)):
            index = sketch_path_index_tracker[test_sketch_paths[ii]]
            test_sketch_x[ii, :] = sketch_features[index, :]

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
        test_image_paths_ext_index_tracker = {}
        for idx in range(len(image_ext_paths)):
            test_image_paths_ext_index_tracker[image_ext_paths[idx]] = idx

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

        return train_sketch_x, test_sketch_x, train_x_img, test_x_img, test_image_paths_ext, test_img_vgg_features_ext

    # 处理数据
    def get_data(self):
        # 加载数据：12500张图片和vgg特征，75479张素描图和vgg特征，增强数据集到73002张图片和vgg特征
        (image_vgg_features, image_paths, sketch_features, sketch_paths,
         image_ext_vgg_features, image_ext_paths, test_split_ref) = self._load_data()

        # 划分数据集：将原始素描图划分为训练集（62785）和测试集（75479 - 62785）
        train_sketch_paths, test_sketch_paths, train_classes, test_classes = self._split_data_path(sketch_paths,
                                                                                                   test_split_ref)

        # 划分特征：训练sketch，测试sketch，训练图片，测试图片，增强数据集中的测试类图片和vgg特征
        train_sketch_x, test_sketch_x, train_x_img, test_x_img, test_image_paths_ext, test_img_vgg_features_ext = \
            self._split_data_features(sketch_paths, image_paths, image_ext_paths, train_sketch_paths, test_sketch_paths,
                                      sketch_features, image_vgg_features, image_ext_vgg_features, train_classes,
                                      self.n_x, self.n_y)

        # 增强数据集中测试图片的类别
        test_image_ext_classes = np.array([path.split('/')[-2] for path in test_image_paths_ext])

        # 原始数据集中测试sketch的类别
        test_sketch_classes = np.array([path.split('/')[-2] for path in test_sketch_paths])

        return (test_image_ext_classes, train_classes, test_sketch_classes,
                test_img_vgg_features_ext, test_image_paths_ext,
                train_sketch_x, train_x_img, test_sketch_paths, test_sketch_x)

    pass


class CVAE(object):

    def __init__(self, sketchy_data, max_epoch=25, batch_size=1024, summary_path="summary"):

        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.summary_path = summary_path

        self.n_x = sketchy_data.n_x
        self.n_y = sketchy_data.n_y

        self.n_z = 1024
        self.hidden_size = 2048

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        KTF.set_session(session=self.sess)

        self.decoder, self.vae = self.build()

        (self.image_classes, self.train_classes, self.test_sketch_classes,
         self.img_vgg_features_ext, self.image_paths_ext,
         self.train_sketch_x, self.train_x_img, self.test_sketch_paths, self.test_sketch_x) = sketchy_data.get_data()

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
            input_combined = concatenate([_image_features, _sketch_features])

            # Construct Encoder
            temp_h_q = Dense(self.hidden_size * 2, activation='relu')(input_combined)
            temp_h_q_bn = BatchNormalization()(temp_h_q)
            h_q_zd = Dropout(rate=0.3)(temp_h_q_bn)
            h_q = Dense(self.hidden_size, activation='relu')(h_q_zd)
            h_q_bn = BatchNormalization()(h_q)

            # parameters of hidden variable
            _mu = Dense(self.n_z, activation='tanh')(h_q_bn)
            _log_sigma = Dense(self.n_z, activation='tanh')(h_q_bn)

            _encoder = Model(inputs=[_sketch_features, _image_features], outputs=[_mu])
            return _mu, _log_sigma, _encoder

        # 解码部分
        def net_decoder(_sketch_features, _input_z):
            d_in = concatenate([_input_z, _sketch_features])
            d_h = decoder_hidden(d_in)
            _d_out = decoder_out(d_h)
            return _d_out

        # 输入
        sketch_features = Input(shape=[self.n_y], name='sketch_features')
        image_features = Input(shape=[self.n_x], name='image_features')
        input_z = Input(shape=[self.n_z], name='input_z')

        # 编码
        self.mu, self.log_sigma, encoder = net_encoder(sketch_features, image_features)

        # 解码层
        decoder_hidden = Dense(self.hidden_size, activation='relu')
        decoder_out = Dense(self.n_x, activation='relu', name='decoder_out')

        # 解码
        d_out = net_decoder(sketch_features, input_z)
        decoder = Model(inputs=[sketch_features, input_z], outputs=[d_out])

        # 隐变量
        sample_z = Lambda(self._sample_z)([self.mu, self.log_sigma])
        # 重构Image、Sketch
        recons_image = net_decoder(sketch_features, sample_z)
        _recons_int = Dense(self.hidden_size, activation='relu')(recons_image)
        recons_sketch = Dense(self.n_y, activation='relu', name='recons_output')(_recons_int)

        # VAE模型
        vae = Model(inputs=[sketch_features, image_features], outputs=[recons_image, recons_sketch])

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

    def test(self, epoch):

        def map_change(input_arr):
            dup = np.copy(input_arr)
            for idx in range(input_arr.shape[1]):
                if idx != 0:
                    dup[:, idx] = dup[:, idx - 1] + dup[:, idx]
            return np.multiply(dup, input_arr)

        # 噪声
        noise_z = np.random.normal(size=[len(self.test_sketch_paths), self.n_z])
        # 测试sketch
        sketch_features_test = np.asarray([self.test_sketch_x[ii].copy() for ii in range(0, len(self.test_sketch_paths))])

        # 预测的图片的重构特征
        # noise + sketch -> image features
        predict_image_features = self.decoder.predict(
            {'sketch_features': sketch_features_test, 'input_z': noise_z}, verbose=2)

        # 200个最近邻
        neigh_num = 200
        # 增强数据集中测试图片的vgg特征
        nearest_neigh = NearestNeighbors(neigh_num, metric='cosine', algorithm='brute').fit(self.img_vgg_features_ext)

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
        print('')
        print('The mean precision@200 for test sketches is ' + str(precision_neigh_num))
        print('The mAP for test_sketches is ' + str(m_ap))

        return precision_neigh_num

    def run(self, model_file, load_weights=True, save_weights=True):

        model_path = os.path.split(model_file)[0]
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # 载入模型
        if os.path.exists(model_file) and load_weights:
            self.vae.load_weights(model_file, skip_mismatch=True)

        tensor_board = callbacks.TensorBoard(log_dir=self.summary_path, histogram_freq=0, update_freq=1000)
        test_callback = callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self.test(epoch + 1))

        # 测试
        self.test(0)

        # 训练
        self.vae.fit(x={'sketch_features': self.train_sketch_x, 'image_features': self.train_x_img},
                     y=[self.train_x_img, self.train_sketch_x],
                     batch_size=self.batch_size, epochs=self.max_epoch, verbose=2,
                     callbacks=[tensor_board, test_callback])

        # 测试
        self.test(self.max_epoch)

        # 保存模型
        if save_weights:
            self.vae.save_weights(model_file)

        pass

    pass


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    name = "sketch_model_2"

    CVAE(sketchy_data=SketchyData(n_x=4096, n_y=4096), max_epoch=300,
         batch_size=1024, summary_path="model/{}/summary".format(name)).run(
        model_file="model/{}/model.h5".format(name), load_weights=False, save_weights=True)
