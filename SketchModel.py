import os
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing import image
from PreStepData import PreStepData, Tools
from keras import models, layers, losses, callbacks, preprocessing
import keras.backend.tensorflow_backend as ktb


class SketchData(object):

    def __init__(self, sketch_data_npy="./data/sketch/sketch_paths_class.npy",
                 sketch_train_image_npy="./data/sketch/sketch_train_image.npy",
                 sketch_test_image_npy="./data/sketch/sketch_test_image.npy",
                 sketch_train_label_npy="./data/sketch/sketch_train_label.npy",
                 sketch_test_label_npy="./data/sketch/sketch_test_label.npy",
                 train_test_ratio=5, pre_process=Tools.pre_process_input, target_size=128, re_data=False):

        # 网络输入大小
        self.input_shape = [target_size, target_size, 1]
        Tools.print("image shape is {}".format(self.input_shape))

        # 如果不存在或者重新生成数据
        if not os.path.exists(sketch_train_image_npy) or re_data:
            # 读取图片路径和标签
            Tools.print("data not exist and produce data...")
            (self._sketch_train_image_path, self._sketch_train_label, self._sketch_test_image_path,
             self._sketch_test_label) = self._read_sketch_path(sketch_data_npy, train_test_ratio)

            # 读取数据
            Tools.print("read data...")
            self._sketch_train_image = self._read_data(self._sketch_train_image_path, pre_process, target_size)
            self._sketch_test_image = self._read_data(self._sketch_test_image_path, pre_process, target_size)

            # 保存
            Tools.print("save data...")
            np.save(sketch_train_image_npy, self._sketch_train_image)
            np.save(sketch_test_image_npy, self._sketch_test_image)
            np.save(sketch_train_label_npy, self._sketch_train_label)
            np.save(sketch_test_label_npy, self._sketch_test_label)
            Tools.print("data save ok")
        else:
            Tools.print("data exist and load data...")
            self._sketch_train_image = np.load(sketch_train_image_npy)
            self._sketch_test_image = np.load(sketch_test_image_npy)
            self._sketch_train_label = np.load(sketch_train_label_npy)
            self._sketch_test_label = np.load(sketch_test_label_npy)
            Tools.print("data load ok")
            pass

        # 类别数
        self.class_num = self._sketch_train_label.max() + 1
        Tools.print("class num is {}".format(self.class_num))
        pass

    @staticmethod
    def _read_sketch_path(sketch_data_npy, train_test_ratio):
        # 获取目录
        if not os.path.exists(sketch_data_npy):
            pre_step_data = PreStepData()
            sketch_paths = pre_step_data.load_sketch_paths()
            sketch_classes = np.asarray([os.path.basename(os.path.split(_)[0]) for _ in sketch_paths])

            test_class = [_.decode() for _ in np.load('./data/ZSSBIR_data/test_split_ref.npy')]
            train_class = list(set(sketch_classes).difference(test_class))

            result_sketch = []
            for sketch_index, sketch_class in enumerate(sketch_classes):
                if sketch_class in train_class:
                    label = train_class.index(sketch_class)
                    result_sketch.append([sketch_paths[sketch_index], sketch_class, label])
                pass
            np.save(sketch_data_npy, result_sketch)
            pass

        Tools.print("read sketch paths")
        result_sketches = np.load(sketch_data_npy)

        Tools.print("split train/test sketch")
        train_data = []
        test_data = []
        for result_sketch in result_sketches:
            if np.random.randint(0, train_test_ratio) == 0:
                test_data.append(result_sketch)
            else:
                train_data.append(result_sketch)
            pass

        sketch_train_image = np.asarray(np.asarray(train_data)[:, 0])
        sketch_train_label = np.asarray(np.asarray(train_data)[:, 2], dtype=np.int32)
        sketch_test_image = np.asarray(np.asarray(test_data)[:, 0])
        sketch_test_label = np.asarray(np.asarray(test_data)[:, 2], dtype=np.int32)

        return sketch_train_image, sketch_train_label, sketch_test_image, sketch_test_label

    @staticmethod
    def _read_data(data_paths, pre_process_input, target_size):
        Tools.print("read data")
        x_data = []
        for data_path_index, data_path in enumerate(data_paths):
            if data_path_index % 1000 == 0:
                Tools.print("{}/{}".format(data_path_index, len(data_paths)))
            _now_data = image.img_to_array(image.load_img(data_path, color_mode="grayscale",
                                                          target_size=(target_size, target_size)))
            x_data.append(_now_data)
            pass
        x_data = np.asarray(x_data)
        x_data = pre_process_input(x_data)

        return x_data

    def get_sketch_train_data(self):
        return self._sketch_train_image, self._sketch_train_label

    def get_sketch_test_data(self):
        return self._sketch_test_image, self._sketch_test_label

    pass


class SketchModel(object):

    def __init__(self, sketch_data, max_epoch, batch_size, summary_path="summary"):

        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.summary_path = summary_path

        # 数据
        self.sketch_train_image, self.sketch_train_label = sketch_data.get_sketch_train_data()
        self.sketch_test_image, self.sketch_test_label = sketch_data.get_sketch_test_data()

        # 模型
        self.sketch_model = self.build(classes_num=sketch_data.class_num, input_shape=sketch_data.input_shape)

        # 配置
        self.sketch_model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0),
                                  loss=losses.sparse_categorical_crossentropy, metrics=['acc'])

        # 会话
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        ktb.set_session(session=self.sess)
        pass

    def build(self, classes_num, input_shape):
        return self.build_sketch_model(classes_num, input_shape)

    @staticmethod
    def build_sketch_model(classes_num, input_shape):
        inputs = layers.Input(shape=input_shape, name='inputs')
        # Block 1
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        # x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        # x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        # Block 6
        x = layers.Conv2D(1024, (3, 3), activation='relu', padding='same', name='block6_conv1')(x)
        # x = layers.Conv2D(1024, (3, 3), activation='relu', padding='same', name='block6_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(x)

        # Block 7
        # x = layers.GlobalAveragePooling2D(name="features")(x)
        x = layers.Flatten(name="features")(x)

        # 全连接，用于分类
        x = layers.Dense(classes_num, activation='softmax', name='predictions')(x)

        model = models.Model(inputs, x, name='sketch_model')
        return model

    def _test(self, info):
        predict = self.sketch_model.predict(self.sketch_test_image, verbose=2, batch_size=self.batch_size)

        top_k = 5
        predict_result_sort = np.argsort(predict)[:, ::-1][:, 0: top_k]  # 先arg,再倒序，再前top_k个。
        is_ok_5 = np.zeros(shape=(top_k,))
        for predict_result_index, predict_result in enumerate(predict_result_sort):
            for i in range(top_k):
                if self.sketch_test_label[predict_result_index] in predict_result[0:i + 1]:
                    is_ok_5[i] += 1
                    pass
            pass
        all_num = len(predict_result_sort)
        is_ok_5_ratio = is_ok_5 / all_num

        Tools.print("{} test result top 1 is {}/{}({})".format(info, is_ok_5[0], all_num, is_ok_5_ratio[0]))
        Tools.print("{} test result top 2 is {}/{}({})".format(info, is_ok_5[1], all_num, is_ok_5_ratio[1]))
        Tools.print("{} test result top 3 is {}/{}({})".format(info, is_ok_5[2], all_num, is_ok_5_ratio[2]))
        Tools.print("{} test result top 4 is {}/{}({})".format(info, is_ok_5[3], all_num, is_ok_5_ratio[3]))
        Tools.print("{} test result top 5 is {}/{}({})".format(info, is_ok_5[4], all_num, is_ok_5_ratio[4]))

        return is_ok_5, all_num, is_ok_5_ratio

    def train(self, model_file, csv_log_file, load_weights=True, save_weights=True):

        model_path = os.path.split(model_file)[0]
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # 载入模型
        if os.path.exists(model_file) and load_weights:
            self.sketch_model.load_weights(model_file, skip_mismatch=True)

        # Callback
        tensor_board = callbacks.TensorBoard(log_dir=self.summary_path, histogram_freq=0, update_freq=1000)
        test_callback = callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self._test(epoch + 1))
        csv_logger = callbacks.CSVLogger(csv_log_file)

        # 测试
        self._test("first")

        # 训练：输入和输出在Model中定义
        # 没有数据增强
        # self.sketch_model.fit(x={'inputs': self.sketch_train_image}, y=[self.sketch_train_label],
        #                       batch_size=self.batch_size, epochs=self.max_epoch,
        #                       verbose=2, callbacks=[tensor_board, csv_logger, test_callback])

        # 数据增强
        data_gen = preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                                                          rotation_range=30, width_shift_range=0.1,
                                                          height_shift_range=0.1, zoom_range=0.2)
        data_gen.fit(self.sketch_train_image)
        self.sketch_model.fit_generator(
            generator=data_gen.flow(self.sketch_train_image, self.sketch_train_label, batch_size=self.batch_size),
            epochs=self.max_epoch, verbose=2, callbacks=[tensor_board, csv_logger, test_callback],
            steps_per_epoch=len(self.sketch_train_image) // self.batch_size)

        # 测试
        self._test("final")

        # 保存模型
        if save_weights:
            self.sketch_model.save_weights(model_file)

        pass

    pass


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    name = "first_aug"
    _sketch_data = SketchData(re_data=False)
    sketch_model = SketchModel(sketch_data=_sketch_data, max_epoch=100,
                               batch_size=128, summary_path="model_sketch/{}/summary".format(name))
    sketch_model.train(model_file="model_sketch/{}/model.h5".format(name),
                       csv_log_file="model_sketch/{}/log.log".format(name),
                       load_weights=False, save_weights=True)
    pass
