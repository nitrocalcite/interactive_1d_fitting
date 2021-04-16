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
        return self._update_params_from_scroll(ticks, self.make_funcargs(params, **kwargs))

    def _update_params_from_scroll(self, ticks, vals):
        """ Describes how parameters should be updated from a scroll event

        Should be implemented by interactive component subclasses
        Args:
            ticks: the number of scroll wheel ticks on this event
            vals: a dictionary describing the current function parameter values for this model
        Returns:
            A Parameters object describing this model for the given selector
        """
        raise NotImplementedError

    def to_poly_selector_shape(self, x_range, params=None, **kwargs) -> Tuple[list, list]:
        return self._to_poly_selector_shape(x_range, self.make_funcargs(params, **kwargs))

    def _to_poly_selector_shape(self, x_range, vals) -> Tuple[list, list]:
        """ Transform given model parameters into a poly selector

        Should be implemented by interactive component subclasses
        Args:
            x_range: a 2-tuple with the min and max values for the independent coordinate
            vals: a dictionary describing the current function parameter values for this model
        Returns:
            Two lists of the same length, describing the x-points and the y-points to draw the polygon at
        """
        raise NotImplementedError

    def from_poly_selector_shape(self, xs, ys):
        return self._from_poly_selector_shape(xs, ys)

    def _from_poly_selector_shape(self, xs, ys):
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
class ILinearModel(InteractiveModelMixin, lmfit.models.LinearModel):
    def _update_params_from_scroll(self, ticks, vals):
        vals['slope'] += ticks/1400
        return self.make_params(**vals)

    def _to_poly_selector_shape(self, x_range, vals):
        return list(x_range), list(self.eval(x=np.array(x_range), **vals))

    def _from_poly_selector_shape(self, xs, ys):
        m = (ys[1] - ys[0])/(xs[1] - xs[0])
        b = ys[0] - (xs[0] * m)
        return self.make_params(slope=m, intercept=b)


class InteractiveDistributionMixin(InteractiveModelMixin):
    def _update_params_from_scroll(self, ticks, vals):
        scale = np.clip((ticks / 700) + 1, 0.5, 2)
        vals['sigma'] *= scale
        vals['amplitude'] *= scale  # don't change peak value
        return self.make_params(**vals)

    def _to_poly_selector_shape(self, x_range, vals):
        c = vals['center']
        s = vals['sigma']
        xs = np.array([c - s, c])  # pick the peak & one sigma to the left of the peak
        ys = self.eval(x=xs, **vals)
        return list(xs), list(ys)

    def _from_poly_selector_shape(self, xs, ys):
        center = xs[-1]
        sigma = abs(xs[-1] - xs[0])
        amplitude = ys[-1] * sigma / self.height_factor
        return self.make_params(amplitude=amplitude, center=center, sigma=sigma)


class IGaussianModel(InteractiveDistributionMixin, lmfit.models.GaussianModel):
    pass


class ILorentzianModel(InteractiveDistributionMixin, lmfit.models.LorentzianModel):
    pass


class IConstantModel(InteractiveModelMixin, lmfit.models.Model):
    """ A ConstantModel that behaves as expected with numpy arrays; see lmfit#684 for details """
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def constant(x, c=0.0):
            return c * np.ones(x.shape)
        super().__init__(constant, **kwargs)

    def guess(self, data, **kwargs):
        pars = self.make_params()
        pars['%sc' % self.prefix].set(value=data.mean())
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    def _update_params_from_scroll(self, ticks, vals):
        vals['c'] += ticks/700
        return self.make_params(**vals)

    def _to_poly_selector_shape(self, x_range, vals):
        return [sum(x_range)/2], [vals['c']]

    def _from_poly_selector_shape(self, xs, ys):
        return self.make_params(c=ys[0])


# composite model isn't a component, so no need to implement any methods.  this is just a preferred change
class ICompositeModel(InteractiveModelMixin, lmfit.model.CompositeModel):
    def guess(self, data, **kws):
        p = lmfit.Parameters()
        p.update(self.left.guess(data, **kws))
        p.update(self.right.guess(data, **kws))
        return p
