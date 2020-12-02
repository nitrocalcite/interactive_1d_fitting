from functools import partial

import lmfit
import param
import panel as pn
import numpy as np

param_fields = ['value', 'min', 'max', 'vary', 'expr', 'brute_step', 'user_data']
max_float = 1e308


def clip(n):
    return np.clip(n, -max_float, max_float)


class InteractiveParameter(param.Parameterized, lmfit.Parameter):
    name = param.String()
    min = param.Number(allow_None=True)
    max = param.Number(allow_None=True)
    vary = param.Boolean()
    _expr = param.String(allow_None=True)

    def __init__(self, *args, **kwargs):
        lmfit.Parameter.__init__(self, *args, **kwargs)
        super().__init__(name=self.name, min=self.min, max=self.max, vary=self.vary)

    @property
    def clipped_min(self):
        """ Tracks `min`, but is JSON-compliant

        This means that it is clipped to +- `max_float`.
        When it is set, values greater or equal to `max_float` are interpreted as infinities.
        If it is set with a NaN value, it will interpret it as -inf
        """
        return clip(self.min)

    @clipped_min.setter
    def clipped_min(self, clipped_min):
        if clipped_min is None:
            self.min = -np.inf
        elif clipped_min >= max_float:
            self.min = np.inf
        elif clipped_min <= -max_float:
            self.min = -np.inf
        else:
            self.min = clipped_min

    @property
    def clipped_max(self):
        """ Tracks `max`, but is JSON-compliant

        See `clipped_min` for details.
        If this is set to a NaN value, it will be interpreted as +inf
        """
        return clip(self.max)

    @clipped_max.setter
    def clipped_max(self, clipped_max):
        if clipped_max is None:  # if you type in an out of range value JS-side, you'll get None here...
            self.max = np.inf
        elif clipped_max >= max_float:
            self.max = np.inf
        elif clipped_max <= -max_float:
            self.max = -np.inf
        else:
            self.max = clipped_max

    @lmfit.Parameter.value.setter
    def value(self, value):
        # this prevents serialization errors
        if abs(value) > max_float:
            raise ValueError(f"Absolute value of {value} must be less than {max_float}")
        lmfit.Parameter.value.fset(self, value)
        # call our callback if assigned when we are set
        if hasattr(self, "on_value_change") and callable(self.on_value_change):
            self.on_value_change(value)

        # ostensibly this should be done with a Dynamic parameter, or an @depends watcher
        # there's an edge case if someone sets self._val, we won't catch that, but _getval() will of course change
        # however, currently, this only happens during __init__ so this callback works well enough
        # (yes, this is fragile wrt updates to lmfit)

    @classmethod
    def from_parameter(cls, parameter):
        """ Factory method to create InteractiveParameter from a Parameter

        Only the name and the fields defined in `param_fields` are preserved.
        """
        o = cls(parameter.name)
        for a in param_fields:
            setattr(o, a, getattr(parameter, a))
        return o


class InteractiveParameters(lmfit.Parameters):
    # noinspection PyTypeChecker
    def controls(self, value_callback=None):
        """ Produces a control Panel for these parameters """
        panel = pn.GridSpec()

        # make the header row
        cols = ["Name", "Value", "Min", "Max", "Vary", "Expr"]
        for n, col in enumerate(cols):
            panel[0, n] = pn.pane.Markdown(f"### {col}", height_policy='min', width_policy='min', width=50)

        num_kws = dict(height=30, height_policy='fit')

        def apply(p, **kws):
            for k, v in kws.items():
                setattr(p, k, v)
            for k in ['width', 'height', 'width_policy', 'height_policy']:
                p.param[k].readonly = True

        # set all non-interactive parameters to interactive ones
        dnc = getattr(self, '_disable_nis_callback', False)
        self._disable_nis_callback = True
        try:
            for k, v in self.items():
                if not isinstance(v, InteractiveParameter):
                    self[k] = InteractiveParameter.from_parameter(v)
        finally:
            self._disable_nis_callback = dnc

        # compose the entire widget grid going row by row.  Each row is one parameter
        for n, (name, lfparam) in enumerate(self.items()):
            n += 1  # row index offset for the header

            # probably should migrate this code to InteractiveParameter, but no pressing reason yet

            # Add the parameter name
            panel[n, 0] = pn.pane.Str(name, width=100, width_policy='fit')

            # Add the parameter value
            value_pane = pn.widgets.FloatInput(value=clip(lfparam.value), width=100, width_policy='min', format='0.00')

            value_pane.link(lfparam, value='value')  # when the GUI updates, set the parameter value
            def value_update(p, v):
                p.value = v

            lfparam.on_value_change = partial(value_update, value_pane)  # when the parameter updates, update the GUI
            if callable(value_callback):
                value_pane.link(lfparam, callbacks={'value': value_callback})  # external callback for any value updates
            panel[n, 1] = value_pane

            # Add the parameter min
            min_pane = pn.widgets.FloatInput(value=clip(lfparam.min), width=100, width_policy='min')
            # when the GUI updates, update the parameter minimum.  interpret out of range values as -inf
            min_pane.link(lfparam, value='clipped_min')
            # when the parameter minimum changes, update the GUI with an in-range value.
            # (so this could be called spuriously, for example if the min goes from -1e308 to -inf)
            def update(p, lfp, *events): p.value = lfp.clipped_min
            lfparam.param.watch(partial(update, min_pane, lfparam), ['min'])
            panel[n, 2] = min_pane

            # Add the parameter max identically to the minimum
            max_pane = pn.widgets.FloatInput(value=clip(lfparam.max), width=100, width_policy='min')
            max_pane.link(lfparam, value='clipped_max')
            def update(p, lfp, *events): p.value = lfp.clipped_max
            lfparam.param.watch(partial(update, max_pane, lfparam), ['max'])
            panel[n, 3] = max_pane

            # Add the parameter vary
            vary_pane = pn.widgets.Checkbox(value=lfparam.vary, width_policy='min', width=50)
            vary_pane.link(lfparam, value='vary', bidirectional=True)  # no funny business as we have two params
            panel[n, 4] = vary_pane

            # Add the parameter expr
            expr_pane = pn.widgets.TextInput(value=lfparam.expr, width=150, width_policy='min')
            expr_pane.link(lfparam, value='expr')  # when the GUI updates, update the parameter value
            def update(p, lfp, *events): p.value = lfp.expr
            # public expr isn't a param, so we watch the plain private expr.
            # however this should be robust, as we are still using values from the public value
            lfparam.param.watch(partial(update, expr_pane, lfparam), ['_expr'])
            panel[n, 5] = expr_pane

            # now do some misc callbacks
            # we don't have a proper watcher on value, but changing expr can change the calculated value
            # so update that
            def update(p, lfp, *events): p.value = lfp.value
            expr_pane.link(lfparam, callbacks={'value': partial(update, value_pane, lfparam)})

            # disable value pane when there's an expression
            def update(ep, panes, *events):
                disabled = not(ep.value == "" or ep.value is None)
                for p in panes:
                    p.disabled = disabled
            expr_pane.link(value_pane, callbacks={'value': partial(update, expr_pane,
                                                                   [value_pane, min_pane, max_pane, vary_pane])})
            # run these misc callbacks
            expr_pane.param.trigger('value')

        for pane in panel:
            apply(pane, **num_kws)
        return panel

    def update_existing_values(self, params):
        """ Updates the value of all existing Parameter objects from supplied `params`

        Does not change the internal Parameter objects
        Does not update the value if an expression is already set
        """
        for c, p in params.items():
            if c in self and self[c].expr is None:
                self[c].value = p.value

    def dumps(self, **kws):
        # we aren't ourselves serializable nicely (callbacks?), so we make a plain non-interactive copy to serialize
        return super().__class__.dumps(self.copy(), **kws)

    def __setitem__(self, key, new):
        if new is not None and not isinstance(new, lmfit.Parameter):
            raise ValueError("'%s' is not a Parameter" % new)

        # this is supposed to make parameters less fragile by making it harder to overwrite the objects directly
        # and calling a callback when we are updated with new, non-interactive parameters
        # I'm not confident that this covers all edge cases; experience would suggest it doesn't, but I can't name any
        # at the moment

        old = self.get(key, None)
        old_int = isinstance(old, InteractiveParameter)
        new_int = isinstance(new, InteractiveParameter)

        if old_int and (not new_int):
            # we are replacing an interactive parameter with a non-interactive one
            # if the names are the same then copy over the attributes
            # no callback needed as we avoid replacing the parameter at all
            if old.name == new.name:
                for f in param_fields:  # this is sort of fragile against new lmfit features...
                    setattr(self[key], f, getattr(new, f))
                return

        super().__setitem__(key, new)
        if hasattr(self, '_disable_nis_callback') and self._disable_nis_callback:
            return
        if hasattr(self, 'on_noninteractive_update') and callable(self.on_noninteractive_update):
            self.on_noninteractive_update()

    def add_many(self, *parlist):
        dnc = getattr(self, '_disable_nis_callback', False)
        self._disable_nis_callback = True
        super().add_many(*parlist)
        self._disable_nis_callback = dnc

        # for performance reasons, just call this once rather than after every __setitem__ as
        # GUI redrawing is pretty expensive.  there's no other reason for this method override
        if hasattr(self, 'on_noninteractive_update') and callable(self.on_noninteractive_update):
            self.on_noninteractive_update()
