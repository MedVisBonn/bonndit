from abc import ABC, abstractmethod

class ReconstModel(ABC):
    def __init__(self, gtab):
        """

        :param gtab:
        """
        self.gtab = gtab

    @abstractmethod
    def fit(self, data, mask, **kwargs):
        """

        :param data:
        :param mask:
        :param kwargs:
        :return:
        """
        msg = "This model does not have fitting implemented yet"
        raise NotImplementedError(msg)


class ReconstFit(ABC):
    def __init__(self, coeffs):
        """

        :param coeffs:
        """
        self.coeffs = coeffs

    @abstractmethod
    def predict(self, gtab):
        """

        :param gtab:
        :return:
        """
        msg = "This model does not have prediction implemented yet"
        raise NotImplementedError(msg)
