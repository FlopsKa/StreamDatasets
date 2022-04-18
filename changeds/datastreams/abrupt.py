import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from skmultiflow.data import led_generator, random_rbf_generator
from tensorflow import keras

from changeds.abstract import ChangeStream, RegionalChangeStream, RandomOrderChangeStream, QuantifiesSeverity
from changeds.helper import har_data_dir, gas_sensor_data_dir

_type = "A"


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

    def type(self) -> str:
        return _type


class RandomOrderMNIST(RandomOrderChangeStream, RegionalChangeStream):
    def __init__(self, num_concepts: int = 100, n_per_concept: int = 2000, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])
        data, y, change_points = RandomOrderChangeStream.create_changes(x, y, n_per_concept=n_per_concept,
                                                                        num_concepts=num_concepts)
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

    def type(self) -> str:
        return _type


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

    def type(self) -> str:
        return _type


class RandomOrderFashionMNIST(RandomOrderChangeStream, RegionalChangeStream):
    def __init__(self, num_concepts: int = 100, n_per_concept: int = 2000, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])
        data, y, change_points = RandomOrderChangeStream.create_changes(x, y, n_per_concept=n_per_concept,
                                                                        num_concepts=num_concepts)
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

    def type(self) -> str:
        return _type


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

    def type(self) -> str:
        return _type

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
    
    def type(self) -> str:
        return _type


class RandomOrderCIFAR10(RandomOrderChangeStream, RegionalChangeStream):
    def __init__(self, num_concepts: int = 100, n_per_concept: int = 2000, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.dot([0.299, 0.587, 0.114])
        x_test = x_test.dot([0.299, 0.587, 0.114])
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train.reshape(-1), y_test.reshape(-1)])
        data, y, change_points = RandomOrderChangeStream.create_changes(x, y, n_per_concept=n_per_concept,
                                                                        num_concepts=num_concepts)
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

    def type(self) -> str:
        return _type


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

    def type(self) -> str:
        return _type


class RandomOrderCIFAR100(RandomOrderChangeStream, RegionalChangeStream):
    def __init__(self, num_concepts: int = 100, n_per_concept: int = 2000, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        x_train = x_train.dot([0.299, 0.587, 0.114])
        x_test = x_test.dot([0.299, 0.587, 0.114])
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train.reshape(-1), y_test.reshape(-1)])
        data, y, change_points = RandomOrderChangeStream.create_changes(x, y, n_per_concept=n_per_concept,
                                                                        num_concepts=num_concepts)
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

    def type(self) -> str:
        return _type


class LED(ChangeStream, RegionalChangeStream, QuantifiesSeverity):

    def __init__(self, n_per_concept: int = 10000, num_concepts: int = 10, has_noise=True, preprocess=None, seed=0):
        """
        Creates a sudden, but
        :param n_per_concept:
        :param num_concepts:
        :param has_noise:
        :param preprocess:
        """
        self.has_noise = has_noise
        random_state = seed
        x = []
        self._invert_probability = [(i + 1) / num_concepts if i % 2 == 1 else 0 for i in range(num_concepts)]
        for i, proba in enumerate(self._invert_probability):
            x.append(led_generator.LEDGenerator(random_state=random_state, has_noise=has_noise,
                                                noise_percentage=proba).next_sample(n_per_concept)[0])
        y = [i for i in range(num_concepts) for _ in range(n_per_concept)]
        x = np.concatenate(x, axis=0)
        if preprocess:
            x = preprocess(x)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        self._invert_probability = [
            prob for prob in self._invert_probability for _ in range(n_per_concept)
        ]
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

    def get_severity(self):
        new_label = self._invert_probability[self.sample_idx]
        old_label = self._invert_probability[self.sample_idx - 2]
        return np.abs(new_label - old_label)

    def type(self) -> str:
        return _type


class HAR(ChangeStream, RegionalChangeStream):
    def __init__(self, preprocess=None):
        test = pd.read_csv(os.path.join(har_data_dir, "test.csv"))
        train = pd.read_csv(os.path.join(har_data_dir, "train.csv"))
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

    def type(self) -> str:
        return _type


class RandomOrderHAR(ChangeStream, RegionalChangeStream):
    def __init__(self, num_concepts: int = 100, n_per_concept: int = 2000, preprocess=None):
        test = pd.read_csv(os.path.join(har_data_dir, "test.csv"))
        train = pd.read_csv(os.path.join(har_data_dir, "train.csv"))
        x = pd.concat([test, train])
        x = x.sort_values(by="Activity")
        y = LabelEncoder().fit_transform(x["Activity"])
        x = x.drop(["Activity", "subject"], axis=1).to_numpy()
        if preprocess:
            x = preprocess(x)
        data, y, change_points = RandomOrderChangeStream.create_changes(x, y, n_per_concept=n_per_concept,
                                                                        num_concepts=num_concepts,
                                                                        shuffle_within_concept=True)
        self._change_points = change_points
        super(RandomOrderHAR, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "HAR"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

    def type(self) -> str:
        return _type


class RBF(ChangeStream, RegionalChangeStream):
    def __init__(self, n_per_concept: int = 10000,
                 num_concepts: int = 10, dims: int = 100,
                 n_centroids: int = 10, add_dims_without_drift=True, preprocess=None, seed=0,
                 random_subspace_size=True):
        self.add_dims_without_drift = add_dims_without_drift
        self.dims = dims
        rng = np.random.default_rng(seed)
        if random_subspace_size:
            total_dims = 2 * dims
            self.dims_drift = rng.integers(low=min(3, total_dims), high=total_dims)
            self.dims_no_drift = total_dims - self.dims_drift
        sample_random_state = rng.integers(0, 100)
        model_random_state = rng.integers(0, 100)
        x = []
        no_drift = []
        for i in range(num_concepts):
            model_random_state += i
            x.append(random_rbf_generator.RandomRBFGenerator(model_random_state=model_random_state,
                                                             sample_random_state=sample_random_state,
                                                             n_features=self.dims_drift,
                                                             n_centroids=n_centroids).next_sample(n_per_concept)[0])
            if add_dims_without_drift:
                no_drift_model_random_state = num_concepts  # a random seed that we will not use to create drifts
                no_drift.append(random_rbf_generator.RandomRBFGenerator(model_random_state=no_drift_model_random_state,
                                                                        sample_random_state=sample_random_state,
                                                                        n_features=self.dims_no_drift,
                                                                        n_centroids=n_centroids
                                                                        ).next_sample(n_per_concept)[0])
        y = [i for i in range(num_concepts) for _ in range(n_per_concept)]
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
        change_dims = np.arange(self.dims_drift)
        return np.asarray([
            change_dims for cp in self.change_points() if cp
        ])

    def type(self) -> str:
        return _type


class GasSensors(RandomOrderChangeStream):
    def __init__(self, num_concepts: int = 100, n_per_concept: int = 2000, preprocess=None):
        df = pd.read_csv(os.path.join(gas_sensor_data_dir, "gas-drift_csv.csv"))
        y = df["Class"].to_numpy()
        x = df.drop("Class", axis=1).to_numpy()
        if preprocess:
            x = preprocess(x)
        data, y, change_points = RandomOrderChangeStream.create_changes(x, y, n_per_concept=n_per_concept,
                                                                        num_concepts=num_concepts,
                                                                        shuffle_within_concept=True)
        self._change_points = change_points
        super(GasSensors, self).__init__(data=data, y=y)

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self.change_points()[self.sample_idx]

    def id(self) -> str:
        return "Gas"

    def type(self) -> str:
        return _type
