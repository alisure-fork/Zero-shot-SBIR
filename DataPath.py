import os
import time
import numpy as np


class Tools(object):

    @staticmethod
    def print(info):
        print("{} {}".format(time.strftime("%H:%M:%S"), info))
        pass

    @staticmethod
    def pre_process_input(x):
        x = 255.0 - x
        x /= 255.0
        return x

    pass


class DataPath(object):

    def __init__(self, sketchy_root="/home/ubuntu/data1.5TB/Sketchy",
                 data_result_root="./data/alisure", image_path="rendered_256x256/256x256/photo/tx_000100000000",
                 sketch_path="rendered_256x256/256x256/sketch/tx_000100000000", edge_path="edge-hed"):
        # path
        self.image_root = os.path.join(sketchy_root, image_path)
        self.sketch_root = os.path.join(sketchy_root, sketch_path)
        self.edge_root = os.path.join(sketchy_root, edge_path)

        # sketch 相关
        self.sketch_paths_npy = os.path.join(data_result_root, "sketch_paths.npy")
        self.sketch_vgg_features_npy = os.path.join(data_result_root, "sketch_vgg_features.npy")
        self.sketch_model_features_npy = os.path.join(data_result_root, "sketch_model_features.npy")

        # 图像相关
        self.image_paths_npy = os.path.join(data_result_root, "image_paths.npy")
        self.image_vgg_features_npy = os.path.join(data_result_root, "image_vgg_features.npy")

        # 边界相关
        self.edge_paths_npy = os.path.join(data_result_root, "edge_paths.npy")
        self.edge_vgg_features_npy = os.path.join(data_result_root, "edge_vgg_features.npy")
        pass

    def load_sketch_paths(self):
        if not os.path.exists(self.sketch_paths_npy):
            _sketch_paths = []
            for path, _, files in os.walk(self.sketch_root):
                for fileName in files:
                    _sketch_paths.append(path + '/' + fileName)
                pass
            np.save(self.sketch_paths_npy, np.array(_sketch_paths))
            pass
        Tools.print("load sketch paths")
        return np.load(self.sketch_paths_npy)

    def load_image_paths(self):
        if not os.path.exists(self.image_paths_npy):
            _image_paths = []
            for path, _, files in os.walk(self.image_root):
                for fileName in files:
                    _image_paths.append(path + '/' + fileName)
                pass
            np.save(self.image_paths_npy, np.array(_image_paths))
            pass
        Tools.print("load image paths")
        return np.load(self.image_paths_npy)

    def load_edge_paths(self):
        if not os.path.exists(self.edge_paths_npy):
            _image_paths = []
            for path, _, files in os.walk(self.edge_root):
                for fileName in files:
                    _image_paths.append(path + '/' + fileName)
                pass
            np.save(self.edge_paths_npy, np.array(_image_paths))
            pass
        Tools.print("load edge paths")
        return np.load(self.edge_paths_npy)

    pass


class Parm(object):

    sketch_model_target_size = 128
    sketch_model_input_shape = [128, 128, 1]
    sketch_model_class_num = 104

    pass
