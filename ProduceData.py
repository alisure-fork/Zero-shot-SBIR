import os
import numpy as np
import tensorflow as tf
from SketchModel import SketchModel
from keras.applications import vgg16
from keras.preprocessing import image
from DataPath import DataPath, Tools, Parm
import keras.backend.tensorflow_backend as ktf


class ModelData(object):

    def __init__(self):
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        ktf.set_session(session=self.sess)
        pass

    @staticmethod
    def get_vgg_model_and_pre_process():
        Tools.print("load vgg model")
        vgg_model = vgg16.VGG16(weights='imagenet', include_top=True)
        vgg_model.layers.pop()
        vgg_model.layers[-1].outbound_nodes = []
        vgg_model.outputs = [vgg_model.layers[-1].output]
        return vgg_model, vgg16.preprocess_input

    @staticmethod
    def get_sketch_model_and_pre_process(classes_num, input_shape, weights):
        Tools.print("load sketch model")
        # 模型
        _sketch_model = SketchModel.build_sketch_model(classes_num, input_shape)
        # 加载权重
        _sketch_model.load_weights(weights)
        # 处理网络
        _sketch_model.layers.pop()
        _sketch_model.layers[-1].outbound_nodes = []
        _sketch_model.outputs = [_sketch_model.layers[-1].output]

        return _sketch_model, Tools.pre_process_input

    @staticmethod
    def predict(model, pre_process_input, x_in_path, save_path,
                batch_size=64, target_size=(224, 224), color_mode="rgb"):
        steps = len(x_in_path) // batch_size
        lefts = len(x_in_path) % batch_size

        Tools.print("begin to predict {}".format(steps))
        result_out = []
        for step in range(steps):
            if step % 20 == 0:
                Tools.print("predict {}/{}".format(step, steps))
            x_in = np.asarray([image.img_to_array(
                image.load_img(x_in_path[step * batch_size + bs],
                               target_size=target_size, color_mode=color_mode)) for bs in range(batch_size)])
            x_in = pre_process_input(x_in)
            x_out = model.predict_on_batch(x_in)
            result_out.extend(x_out)
            pass

        if lefts > 0:
            Tools.print("predict {}".format(steps + 1))
            x_in = np.asarray([image.img_to_array(
                image.load_img(x_in_path[steps * batch_size + left],
                               target_size=target_size, color_mode=color_mode)) for left in range(lefts)])
            x_in = pre_process_input(x_in)
            x_out = model.predict_on_batch(x_in)
            result_out.extend(x_out)
            pass

        Tools.print("save features {} in {}".format(len(result_out), save_path))
        np.save(save_path, result_out)
        pass

    pass


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 加载路径
    data_path = DataPath()
    sketch_paths = data_path.load_sketch_paths()
    image_paths = data_path.load_image_paths()

    model_data = ModelData()
    # 使用VGG模型处理Image和Sketch数据
    # vgg_model, pre_process = model_data.get_vgg_model_and_pre_process()
    # model_data.predict(model=vgg_model, pre_process_input=pre_process, x_in_path=sketch_paths,
    #                    save_path=pre_step_data.sketch_vgg_features_npy, batch_size=64)
    # model_data.predict(model=vgg_model, pre_process_input=pre_process, x_in_path=image_paths,
    #                    save_path=pre_step_data.image_vgg_features_npy, batch_size=64)

    # 使用SketchModel处理Sketch数据
    sketch_model, pre_process = model_data.get_sketch_model_and_pre_process(
        classes_num=Parm.sketch_model_class_num, input_shape=Parm.sketch_model_input_shape,
        weights="./model_sketch/first_aug/model.h5")
    model_data.predict(model=sketch_model, pre_process_input=pre_process, x_in_path=sketch_paths,
                       save_path=data_path.sketch_model_features_npy, batch_size=64, color_mode="grayscale",
                       target_size=[Parm.sketch_model_target_size, Parm.sketch_model_target_size])
