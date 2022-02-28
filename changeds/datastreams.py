import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import numpy as np

from skmultiflow.data import led_generator, random_rbf_generator

from changeds.abstract import ChangeStream, RegionalChangeStream, ClassificationStream, RandomOrderChangeStream
from changeds.helper import preprocess_hipe


class SortedMNIST(ChangeStream, RegionalChangeStream):
    def __init__(self, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])
        sorted_indices = np.argsort(y)
        x = x[sorted_indices]
        y = y[sorted_indices]
        if preprocess:
            x = preprocess(x)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(SortedMNIST, self).__init__(data=x, y=y)

    def id(self) -> str:
        return "sMNIST"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class RandomOrderMNIST(RandomOrderChangeStream):
    def __init__(self, num_changes: int = 100, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])
        data, y, change_points = RandomOrderChangeStream.create_changes(x, y, num_changes)
        self._change_points = change_points
        if preprocess:
            data = preprocess(data)
        super(RandomOrderMNIST, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "MNIST"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class SortedFashionMNIST(ChangeStream, RegionalChangeStream):
    def __init__(self, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])
        sorted_indices = np.argsort(y)
        x = x[sorted_indices]
        y = y[sorted_indices]
        if preprocess:
            x = preprocess(x)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(SortedFashionMNIST, self).__init__(data=x, y=y)

    def id(self) -> str:
        return "sFMNIST"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class RandomOrderFashionMNIST(RandomOrderChangeStream, RegionalChangeStream):
    def __init__(self, num_changes: int = 100, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])
        data, y, change_points = RandomOrderChangeStream.create_changes(x, y, num_changes)
        self._change_points = change_points
        if preprocess:
            data = preprocess(data)
        super(RandomOrderFashionMNIST, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "FMNIST"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class SortedCIFAR10(ChangeStream, RegionalChangeStream):
    def __init__(self, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.dot([0.299, 0.587, 0.114])
        x_test = x_test.dot([0.299, 0.587, 0.114])
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train.reshape(-1), y_test.reshape(-1)])
        sorted_indices = np.argsort(y)
        x = x[sorted_indices]
        y = y[sorted_indices]

        if preprocess:
            x = preprocess(x)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(SortedCIFAR10, self).__init__(data=x, y=y)

    def id(self) -> str:
        return "sCIFAR"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class SortedCIFAR10_RGB(ChangeStream, RegionalChangeStream):
    def __init__(self, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.reshape((len(x_train), x_train.shape[1] * x_train.shape[2] * x_train.shape[3]))
        x_test = x_test.reshape((len(x_test), x_test.shape[1] * x_test.shape[2] * x_test.shape[3]))
     
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train.reshape(-1), y_test.reshape(-1)])
        sorted_indices = np.argsort(y)
        x = x[sorted_indices]
        y = y[sorted_indices]

        if preprocess:
            x = preprocess(x)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(SortedCIFAR10_RGB, self).__init__(data=x, y=y)

    def id(self) -> str:
        return "sCIFAR_RGB"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class RandomOrderCIFAR10(RandomOrderChangeStream, RegionalChangeStream):
    def __init__(self, num_changes: int = 100, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.dot([0.299, 0.587, 0.114])
        x_test = x_test.dot([0.299, 0.587, 0.114])
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train.reshape(-1), y_test.reshape(-1)])
        data, y, change_points = RandomOrderChangeStream.create_changes(x, y, num_changes)
        self._change_points = change_points
        if preprocess:
            data = preprocess(data)
        super(RandomOrderCIFAR10, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "CIFAR"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class SortedCIFAR100(ChangeStream, RegionalChangeStream):
    def __init__(self, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        x_train = x_train.dot([0.299, 0.587, 0.114])
        x_test = x_test.dot([0.299, 0.587, 0.114])
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train.reshape(-1), y_test.reshape(-1)])
        sorted_indices = np.argsort(y)
        x = x[sorted_indices]
        y = y[sorted_indices]

        if preprocess:
            x = preprocess(x)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(SortedCIFAR100, self).__init__(data=x, y=y)

    def id(self) -> str:
        return "sCIFAR100"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class RandomOrderCIFAR100(RandomOrderChangeStream, RegionalChangeStream):
    def __init__(self, num_changes: int = 100, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        x_train = x_train.dot([0.299, 0.587, 0.114])
        x_test = x_test.dot([0.299, 0.587, 0.114])
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train.reshape(-1), y_test.reshape(-1)])
        data, y, change_points = RandomOrderChangeStream.create_changes(x, y, num_changes)
        self._change_points = change_points
        if preprocess:
            data = preprocess(data)
        super(RandomOrderCIFAR100, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "CIFAR100"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class HIPE(ChangeStream):
    def __init__(self, preprocess=None):
        x = preprocess_hipe()
        y = np.zeros(shape=len(x))
        if preprocess:
            x = preprocess(x)
        self._change_points = y
        super(HIPE, self).__init__(data=x, y=y)

    def id(self) -> str:
        return "HIPE"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class LED(ChangeStream, RegionalChangeStream):

    def __init__(self, n_per_concept: int = 10000, n_drifts: int = 10, has_noise=True, preprocess=None):
        """
        Creates a sudden, but
        :param n_per_concept:
        :param n_drifts:
        :param has_noise:
        :param preprocess:
        """
        self.has_noise = has_noise
        random_state = 0
        x = []
        for i in range(n_drifts):
            x.append(led_generator.LEDGenerator(random_state=random_state, has_noise=has_noise,
                                                noise_percentage=(i + 1) / n_drifts if i % 2 == 1 else 0
                                                ).next_sample(n_per_concept)[0])
        y = [i for i in range(n_drifts) for _ in range(n_per_concept)]
        x = np.concatenate(x, axis=0)
        if preprocess:
            x = preprocess(x)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(LED, self).__init__(data=x, y=np.array(y))

    def id(self) -> str:
        return "LED"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self.change_points()[self.sample_idx]

    def approximate_change_regions(self):
        change_dims = np.arange(7)
        return np.asarray([
            change_dims for cp in self.change_points() if cp
        ])


class HAR(ChangeStream, RegionalChangeStream):
    def __init__(self, preprocess=None):
        this_dir, _ = os.path.split(__file__)
        path_to_data = os.path.join(this_dir, "..", "data", "har")
        test = pd.read_csv(os.path.join(path_to_data, "test.csv"))
        train = pd.read_csv(os.path.join(path_to_data, "train.csv"))
        x = pd.concat([test, train])
        x = x.sort_values(by="Activity")
        y = LabelEncoder().fit_transform(x["Activity"])
        x = x.drop(["Activity", "subject"], axis=1)
        if preprocess:
            x = preprocess(x)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(HAR, self).__init__(data=x, y=y)

    def id(self) -> str:
        return "sHAR"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class RandomOrderHAR(ChangeStream, RegionalChangeStream):
    def __init__(self, num_changes: int = 100, preprocess=None):
        this_dir, _ = os.path.split(__file__)
        path_to_data = os.path.join(this_dir, "..", "data", "har")
        test = pd.read_csv(os.path.join(path_to_data, "test.csv"))
        train = pd.read_csv(os.path.join(path_to_data, "train.csv"))
        x = pd.concat([test, train])
        x = x.sort_values(by="Activity")
        y = LabelEncoder().fit_transform(x["Activity"])
        x = x.drop(["Activity", "subject"], axis=1).to_numpy()
        if preprocess:
            x = preprocess(x)
        data, y, change_points = RandomOrderChangeStream.create_changes(x, y, num_changes, shuffle_within_concept=True)
        self._change_points = change_points
        super(RandomOrderHAR, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "HAR"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class RBF(ChangeStream, RegionalChangeStream):
    def __init__(self, n_per_concept: int = 10000,
                 n_drifts: int = 10, dims: int = 100,
                 n_centroids: int = 10, add_dims_without_drift=True, preprocess=None):
        self.add_dims_without_drift = add_dims_without_drift
        sample_random_state = 0
        x = []
        no_drift = []
        for i in range(n_drifts):
            model_random_state = i
            x.append(random_rbf_generator.RandomRBFGenerator(model_random_state=model_random_state,
                                                             sample_random_state=sample_random_state, n_features=dims,
                                                             n_centroids=n_centroids).next_sample(n_per_concept)[0])
            if add_dims_without_drift:
                no_drift_model_random_state = n_drifts  # a random seed that we will not use to create drifts
                no_drift.append(random_rbf_generator.RandomRBFGenerator(model_random_state=no_drift_model_random_state,
                                                                        sample_random_state=sample_random_state,
                                                                        n_features=dims, n_centroids=n_centroids
                                                                        ).next_sample(n_per_concept)[0])
        y = [i for i in range(n_drifts) for _ in range(n_per_concept)]
        x = np.concatenate(x, axis=0)
        if add_dims_without_drift:
            noise = np.concatenate(no_drift, axis=0)
            x = np.concatenate([x, noise], axis=1)
        if preprocess:
            x = preprocess(x)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(RBF, self).__init__(data=x, y=np.array(y))

    def id(self) -> str:
        return "RBF"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self.change_points()[self.sample_idx]

    def approximate_change_regions(self):
        num_change_dims = len(self.y) / 2 if self.add_dims_without_drift else len(self.y)
        change_dims = np.arange(num_change_dims)
        return np.asarray([
            change_dims for cp in self.change_points() if cp
        ])


class ArtificialStream(ClassificationStream):
    def id(self) -> str:
        return self.filename[:-4]

    def __init__(self, filename: str):
        self.filename = filename
        path, _ = os.path.split(__file__)
        path = os.path.join(path, "..", "concept-drift-datasets-scikit-multiflow", "artificial")
        file_path = os.path.join(path, filename)
        assert os.path.exists(file_path), "The requested file does not exist in {}".format(file_path)
        super(ArtificialStream, self).__init__(data_path=file_path)


class RealWorldStream(ClassificationStream):
    def id(self) -> str:
        return self.filename[:-4]

    def __init__(self, filename: str):
        self.filename = filename
        path, _ = os.path.split(__file__)
        path = os.path.join(path, "..", "concept-drift-datasets-scikit-multiflow", "real-world")
        file_path = os.path.join(path, filename)
        assert os.path.exists(file_path), "The requested file does not exist in {}".format(file_path)
        super(RealWorldStream, self).__init__(data_path=file_path)


if __name__ == '__main__':
    stream = RandomOrderHAR()
    print(stream.id())
    while stream.has_more_samples():
        x, y, is_change = stream.next_sample()
        if is_change:
            print("Change at index {}".format(stream.sample_idx))

    if isinstance(stream, RegionalChangeStream):
        change_regions = stream.approximate_change_regions()
        print(change_regions)
