from abc import ABC, abstractmethod
from typing import Tuple
import operator

import lmfit
import numpy as np


class InteractiveModelMixin(lmfit.models.Model, ABC):
    def interactive_guess(self, data, x=None, params=None, plot_xs=None):
        """ Like `guess`, but with an attached interactive front-end """
        from .interactive_guess import InteractiveGuessSession
        igs = InteractiveGuessSession(self, data, x=x, params=params, plot_xs=plot_xs)
        igs.start()
        return igs

    def update_params_from_scroll(self, ticks, params=None, **kwargs):
        """ Describes how parameters should be updated from a scroll event

        Should be implemented by interactive component subclasses
        Args:
            ticks: the number of scroll wheel ticks on this event
            params: an optional Parameters object describing current Parameters
            **kwargs: extra Parameter values that should override those in `params`
        Returns:
            A Parameters object describing the new Parameters that should be applied to this model
        """
        raise NotImplementedError

    def to_poly_selector_shape(self, params=None, **kwargs) -> Tuple[list, list]:
        """ Transform given model parameters into a poly selector

        Should be implemented by interactive component subclasses
        Args:
            params: an optional Parameters object describing current Parameters
            **kwargs: extra Parameter values that should override those in `params`
        Returns:
            Two lists of the same length, describing the x-points and the y-points to draw the polygon at
        """
        raise NotImplementedError

    def from_poly_selector_shape(self, xs, ys):
        """ Transforms a poly selector to parameters for this model

        Should be implemented by interactive component subclasses
        Args:
            xs: the x-coordinate data for a polygon selector
            ys: the y-coordinate data for a polygon selector
        Returns:
            A Parameters object describing this model for the given selector
        """
        raise NotImplementedError

    def __add__(self, other):
        """+"""
        return ICompositeModel(self, other, operator.add)

    def __sub__(self, other):
        """-"""
        return ICompositeModel(self, other, operator.sub)

    def __mul__(self, other):
        """*"""
        return ICompositeModel(self, other, operator.mul)

    def __div__(self, other):
        """/"""
        return ICompositeModel(self, other, operator.truediv)

    def __truediv__(self, other):
        """/"""
        return ICompositeModel(self, other, operator.truediv)


# example of what a model component needs to implement to be interactive
class IGaussianModel(InteractiveModelMixin, lmfit.models.GaussianModel):
    def update_params_from_scroll(self, ticks, params=None, **kwargs):
        vals = self.make_funcargs(params, **kwargs)
        scale = np.clip((ticks / 700) + 1, 0.5, 2)
        vals['sigma'] *= scale
        vals['amplitude'] *= scale  # don't change peak value
        return self.make_params(**vals)

    def to_poly_selector_shape(self, params=None, **kwargs):
        vals = self.make_funcargs(params, kwargs)
        c = vals['center']
        s = vals['sigma']
        xs = np.array([c - s, c])  # pick the peak & one sigma to the left of the peak
        ys = self.eval(x=xs, **vals)
        return list(xs), list(ys)

    def from_poly_selector_shape(self, xs, ys):
        center = xs[-1]
        sigma = abs(xs[-1] - xs[0])
        amplitude = ys[-1] * np.sqrt(2 * np.pi) * sigma
        return self.make_params(amplitude=amplitude, center=center, sigma=sigma)


# composite model isn't a component, so no need to implement any methods.  this is just a preferred change
class ICompositeModel(InteractiveModelMixin, lmfit.model.CompositeModel):
    def guess(self, data, **kws):
        p = lmfit.Parameters()
        p.update(self.left.guess(data, **kws))
        p.update(self.right.guess(data, **kws))
        return p
