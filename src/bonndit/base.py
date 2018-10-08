from abc import ABC, abstractmethod


class ReconstModel(ABC):
    def __init__(self, gtab):
        """ An abstract base class for DWI models

        Parameters
        ----------
        gtab : dipy.data.GradientTable
            An object holding information about the applied Gradients including
            b-values and b-vectors
        """
        self.gtab = gtab

    @abstractmethod
    def fit(self, data, mask, **kwargs):
        """ An abstract method for fitting the specified model with given data

        Parameters
        ----------
        data : ndarray
            Diffusion Weighted Data. N measurements for every voxel
        mask : ndarray
            Mask specifying all voxels for which to fit the model
        kwargs :
            Keyword Arguments which may differ for each model

        Returns
        -------
        MultiVoxel
            A container object managing the Fit Objects for every voxel

        """
        msg = "This model does not have fitting implemented yet"
        raise NotImplementedError(msg)


class ReconstFit(ABC):
    def __init__(self, coeffs):
        """ An abstract base class for the fits of DWI models

        Parameters
        ----------
        coeffs : ndarray
             Coefficients of the fitted model
        """
        self.coeffs = coeffs

    @abstractmethod
    def predict(self, gtab):
        """ An abstract method for predicting DWI signals given a fitted model

        Parameters
        ----------
        gtab : dipy.data.GradientTable
            Gradients for which to predict the signal.

        Returns
        -------
        ndarray
            Predicted signals

        """
        msg = "This model does not have prediction implemented yet"
        raise NotImplementedError(msg)
