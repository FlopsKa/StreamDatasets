from abc import abstractmethod, ABCMeta, ABC
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from skmultiflow.data import DataStream


class ChangeStream(DataStream, metaclass=ABCMeta):
    def next_sample(self, batch_size=1):
        change = self._is_change()
        x, y = super(ChangeStream, self).next_sample(batch_size)
        return x, y, change

    @abstractmethod
    def change_points(self):
        raise NotImplementedError

    @abstractmethod
    def _is_change(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def id(self) -> str:
        raise NotImplementedError

    def type(self) -> str:
        raise NotImplementedError


@runtime_checkable
class RegionalChangeStream(Protocol):
    def approximate_change_regions(self):
        change_indices = [i for i, is_change in enumerate(self.change_points()) if is_change == 1]
        change_regions = []
        for i, ci in enumerate(change_indices):
            concept_a = self.data[:ci] if i == 0 else self.data[change_indices[i - 1]:ci]
            concept_b = self.data[ci:change_indices[i + 1]] if i < len(change_indices) - 1 else self.data[
                                                                                                ci:len(self.data)]
            mean_concept_a = np.mean(concept_a, axis=0)
            mean_concept_b = np.mean(concept_b, axis=0)
            diff = np.abs(mean_concept_b - mean_concept_a)
            thresholded = diff > 0.5
            change_regions.append([
                i for i, change in enumerate(thresholded) if change
            ])
        return change_regions


class ClassificationStream(ChangeStream, ABC):
    def __init__(self, data_path: str, preprocess=None):
        """
        Use this kind of stream to train a classifier with the data and the label
        - can be useful to evaluate the performance of unsupervised change detection for changes in P(X) && P(y|X)
        :param data_path: path to the csv to load
        :param preprocess: preprocessing function
        """
        data = pd.read_csv(data_path)
        y = data["target"].to_numpy()
        x = data.drop("target", axis=1)
        if preprocess:
            x = preprocess(x)
        super(ClassificationStream, self).__init__(data=x, y=y)

    def change_points(self):
        return np.array([np.nan for _ in range(len(self.y))])

    def _is_change(self) -> bool:
        return False


class RandomOrderChangeStream(ChangeStream, ABC):
    @staticmethod
    def create_changes(X, y, num_concepts: int, shuffle_within_concept: bool = False):
        sorted_indices = np.argsort(y)
        diffs = np.diff(y[sorted_indices], prepend=y[sorted_indices][0]).astype(int)
        new_concept_indices = [i for i in range(len(diffs)) if diffs[i] == 1]
        concepts = np.split(sorted_indices, new_concept_indices)
        if shuffle_within_concept:
            [np.random.shuffle(concept) for concept in concepts]
        drift_points = []
        x_final = []
        y_final = []
        for concept_index in range(num_concepts):
            random_concept = np.random.choice(range(len(concepts)))
            random_concept = concepts[random_concept]
            np.random.shuffle(random_concept)
            x_final.append(X[random_concept])
            y_final.append(y[random_concept])
            if concept_index == num_concepts - 1:
                break
            if len(drift_points) > 0:
                drift_points.append(drift_points[-1] + len(random_concept))
            else:
                drift_points.append(len(random_concept))
        x_final = np.concatenate(x_final, axis=0)
        y_final = np.concatenate(y_final, axis=0)
        change_points = np.zeros_like(y_final)
        drift_points = np.asarray(drift_points).astype(int)
        change_points[drift_points] = 1
        return x_final, y_final, change_points

    def change_points(self):
        pass

    def _is_change(self) -> bool:
        pass


class GradualChangeStream(ChangeStream, ABC):

    def __init__(self, X, y, num_concepts: int = 100, drift_length: int = 100, stretch: bool = False,
                 shuffle_within_concept: bool = False, preprocess = False):
        self.stretch = stretch
        self.dl = drift_length
        self.num_concepts = num_concepts
        X, y, change_points = GradualChangeStream.create_changes(X, y, num_concepts=num_concepts,
                                                                 drift_length=drift_length, stretch=stretch,
                                                                 shuffle_within_concept=shuffle_within_concept)
        self._change_points = change_points
        if preprocess:
            X = preprocess(X)
        super(GradualChangeStream, self).__init__(data=X, y=y)

    def drift_lengths(self) -> np.ndarray:
        if not self.stretch:
            return np.asarray([self.dl for _ in range(self.num_concepts)], dtype=int)
        else:
            return np.asarray([
                (i + 1) / self.num_concepts * self.dl for i in range(self.num_concepts)
            ], dtype=int)

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

    def type(self) -> str:
        return "G"

    @staticmethod
    def create_changes(X, y, num_concepts: int, drift_length: int, stretch: bool, shuffle_within_concept: bool = False):
        """
        Creates gradual changes
        :param X: The data
        :param y: The labels
        :param num_concepts: Number of changes
        :param drift_length: The length of the drift
        :param stretch: If true, the drifts have varying length, the first drift is abrupt, the last drift has 'drift_length'
        :param shuffle_within_concept: If the data within a concept gets shuffled
        :return:
        """
        sorted_indices = np.argsort(y)
        diffs = np.diff(y[sorted_indices], prepend=y[sorted_indices][0]).astype(int)
        new_concept_indices = [i for i in range(len(diffs)) if diffs[i] == 1]
        concepts = np.split(sorted_indices, new_concept_indices)
        if shuffle_within_concept:
            [np.random.shuffle(concept) for concept in concepts]
        drift_points = []
        data_stream_indices = []
        for concept_index in range(num_concepts):
            this_drift_length = (concept_index + 1) / num_concepts * drift_length if stretch else drift_length
            random_concept = np.random.choice(range(len(concepts)))
            random_concept = concepts[random_concept]
            np.random.shuffle(random_concept)
            if len(data_stream_indices) == 0:
                data_stream_indices = random_concept.tolist()
            else:
                if this_drift_length >= 2:
                    half_drift_length = int(this_drift_length / 2)
                    tail = data_stream_indices[-half_drift_length:]
                    head = random_concept[half_drift_length:]
                    for i in range(half_drift_length):
                        if np.random.uniform() < (i + 1) / this_drift_length:
                            current_index = half_drift_length - i
                            t = tail[-current_index]
                            h = head[current_index]
                            data_stream_indices[-current_index] = h
                            random_concept[current_index] = t
                data_stream_indices += random_concept.tolist()
            if concept_index == num_concepts - 1:
                break
            if len(drift_points) > 0:
                drift_points.append(drift_points[-1] + len(random_concept))
            else:
                drift_points.append(len(random_concept))
        x_final = X[data_stream_indices]
        y_final = y[data_stream_indices]
        change_points = np.zeros_like(y_final)
        drift_points = np.asarray(drift_points).astype(int)
        change_points[drift_points] = 1
        return x_final, y_final, change_points


@runtime_checkable
class QuantifiesSeverity(Protocol):
    def get_severity(self):
        raise NotImplementedError
