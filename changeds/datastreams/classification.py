import os

from changeds.abstract import ClassificationStream


class ArtificialStream(ClassificationStream):
    def id(self) -> str:
        return self.filename[:-4]

    def __init__(self, filename: str):
        self.filename = filename
        path, _ = os.path.split(__file__)
        path = os.path.join(path, "..", "..", "concept-drift-datasets-scikit-multiflow", "artificial")
        file_path = os.path.join(path, filename)
        assert os.path.exists(file_path), "The requested file does not exist in {}".format(file_path)
        super(ArtificialStream, self).__init__(data_path=file_path)


class RealWorldStream(ClassificationStream):
    def id(self) -> str:
        return self.filename[:-4]

    def __init__(self, filename: str):
        self.filename = filename
        path, _ = os.path.split(__file__)
        path = os.path.join(path, "..", "..", "concept-drift-datasets-scikit-multiflow", "real-world")
        file_path = os.path.join(path, filename)
        assert os.path.exists(file_path), "The requested file does not exist in {}".format(file_path)
        super(RealWorldStream, self).__init__(data_path=file_path)