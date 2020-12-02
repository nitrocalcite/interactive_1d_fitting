from datetime import datetime
from functools import partial

import numpy as np
import panel as pn
import bokeh.models as bm
import bokeh.events as be
from bokeh.plotting import figure

from .parameter import InteractiveParameters
from .models import InteractiveModelMixin


class InteractiveGuessSession:
    def get_selected_component(self):
        """ Returns the Model object associated with the selected component """
        desired_component_label = self._selector_pane.value
        for component in self._model.components:
            # just return the first one...there shouldn't be duplicates, in theory
            if component.prefix + component.name == desired_component_label:
                return component
        else:
            return None

    def redraw(self):
        """ Redraw all lines on the plot """
        # appropriately update the components
        self._fig_change(self._redraw_components)

        # update the polygon selector
        self.draw_selector_for_component(None, None)

        # this is a hack to make the GUI values that are set by expressions update
        # in order to make this nice, the `value` machinery of `lmfit.Parameter` needs to be rewritten
        # with `param` and `param.depends`, so in the mean time, this is good enough.
        for p in self._params.values():
            if p.expr is not None and p.expr != "":
                p.value = p.value

    def _redraw_components(self):
        """ Redraw all components of this model and the model itself """
        for c, v in self._model.eval_components(params=self._params, x=self.plot_xs).items():
            self._comp_dss[c].data = {'x': self.plot_xs, 'y': v}

        ymodel = self._model.eval(params=self._params, x=self._x)
        self._model_ds.data = {'x': self._x, 'y': ymodel}
        self._res_ds.data = {'x': self._x, 'y': ymodel - self._data}

    def _fig_change(self, f):
        """ Safely call a function f that updates the Bokeh figure state """
        # if the figure isn't yet displayed, `document` is None and we just call the function
        # but if the figure is being displayed, we have to schedule the function execution with the server's IOLoop
        if self._fig.document is None:
            f()
        else:
            self._fig.document.add_next_tick_callback(f)

    def _set_selector_options(self):
        """ Set the options for the component selector """
        self._selector_pane.options = [c.prefix + c.name for c in self._model.components
                                       if isinstance(c, InteractiveModelMixin)] + ['None']

    def set_model(self, model, params=None):
        """ Sets the Model to be fit

        If the Model or its components are not interactive, this interface will still work, but the non-interactive
        components cannot be edited dragging the plot.
        Args:
            model: an interactive Model or CompositeModel instance
            params: a Parameters object to be used as an initial guess for the model
                if None, will use `model.guess` as default setting
        """

        self._model = model
        self._set_selector_options()
        self._set_params(params)

        # hide all existing data sources; Bokeh doesn't allow us to remove them...
        # this is probably a lag issue if you change the model too many times
        def hide_ds(rds):
            rds.data = {'x': [], 'y': []}

        for c, remove_ds in self._comp_dss.items():
            self._fig_change(partial(hide_ds, remove_ds))

        # draw all the components and keep all data sources & renderers
        self._comp_dss = {}
        self._comp_grs = {}

        def render_comp(rds, rc):
            self._comp_grs[rc] = self._fig.line('x', 'y', source=rds, line_dash='dashed')

        for c, v in model.eval_components(params=self._params, x=self.plot_xs).items():
            assert self.plot_xs.shape == getattr(v, 'shape', None), f"Given `self.plot_xs`, model component '{c}' was "\
                                                                    f"expected to evaluate to a numpy array of shape "\
                                                                    f"{self.plot_xs.shape} but it did not; cannot plot"
            ds = bm.ColumnDataSource({'x': self.plot_xs, 'y': v})
            self._fig_change(partial(render_comp, ds, c))
            self._comp_dss[c] = ds

        # we really only need to update the model, but this is fine
        self.redraw()

    def set_params(self, params):
        self._set_params(params)
        if self._params is not None:
            self.redraw()

    def _set_params(self, params):
        """ Private `set_params` that does not redraw """
        if params is None:
            if hasattr(self._model, 'guess') and callable(self._model.guess):
                self._params = self._model.guess(self._data, x=self._x)
            else:
                self._params = None
        else:
            self._params = params

        if self._params is not None:
            self._params.__class__ = InteractiveParameters
            self._params.on_noninteractive_update = lambda *args: self.set_params(self._params)
            self._param_pane = self._params.controls(value_callback=lambda *args: self.redraw())
            self._rhs_pane[-1] = self._param_pane

            # Clear the selector's default value
            self._selector_pane.value = 'None'

    def set_data(self, data, x=None):
        """ Sets the data to fit to the Model

        Args:
             data: the dependent variable data to fit to the model, as a 1D array
             x: the independent variable data to use in the fit.
                if None, will number the data points sequentially
        """
        self._set_data(data, x=x)
        self.redraw()

    def _set_data(self, data, x=None):
        """ Private `set_data` that does not redraw after update, for use in __init__ """
        x = np.arange(data.shape[0]) if x is None else x
        self._data = data
        self._x = x

        def update_data(): self._data_ds.data = {'x': self._x, 'y': self._data}

        self._fig_change(update_data)

    def __init__(self, model, data, params=None, x=None, plot_xs=None):
        """ Allows 1D models to be fit interactively

        The update architecture is this:

            * When the parameters are updated, the plot redraws
            * When the table updates, the parameters are updated
            * When the plot is dragged, the parameters are updated
            * When the mouse is scrolled, the parameters are updated
        Args:
             model: the Model instance to fit
             data: the data to fit the model to
        Keyword Args:
             params: the initial parameters to draw the Model with.  If None, use parameters given by `model.guess`
             x: the x-axis data points for `data`.  If None, will number the data points sequentially
             plot_xs: the x-axis data points to evaluate the model and its components at.  If None, will use `x`
        """
        self._last_fit_result = None
        self._last_fit_result_time = None

        self._last_selector_pane_value = "None"

        # the bokeh figure
        self._fig = fig = figure()

        # set up the figure for the residuals
        self._resfig = figure()
        self._res_ds = bm.ColumnDataSource({'x': [], 'y': []})
        self._resfig.circle_dot('x', 'y', source=self._res_ds, fill_color='red', line_color=None)

        # --- Compose the main Panel application ---
        self._panel = pn.Row()
        _bokeh_pane = pn.pane.Bokeh(fig)
        self._panel.append(pn.Tabs(('Main Plot', pn.Column(_bokeh_pane,
                                                           pn.pane.Markdown("### Residuals"),
                                                           self._resfig)),
                                   ('Residuals Only', self._resfig)))
        self._tabs_pane = pn.Tabs()
        self._panel.append(self._tabs_pane)
        # add the component-edit selector, along with a placeholder for the param editor table
        self._selector_pane = pn.widgets.Select(name="Editing Component", options=['None'])
        self._rhs_pane = pn.Column(self._selector_pane, pn.pane.Str("param pane placeholder"))
        self._tabs_pane.append(("Edit", self._rhs_pane))
        # add the fit button
        self._fit_pane = pn.widgets.Button(name="Fit")
        self._fit_results_pane = pn.pane.Str()
        self._tabs_pane.append(("Fit", pn.Column(self._fit_pane, self._fit_results_pane)))

        # --- Setup & draw functional internal components ---
        # setup the polygon selector tools & renderers for drawing
        self._pdt_ds = bm.ColumnDataSource({'x': [], 'y': []})
        pdt_renderer = fig.multi_line('x', 'y', source=self._pdt_ds, line_color='green')
        pdt = bm.PolyEditTool(renderers=[pdt_renderer], vertex_renderer=fig.circle([], [], size=10, color='green'))
        fig.add_tools(pdt)
        # setup raw data & the independent variable x
        self._data_ds = bm.ColumnDataSource({'x': [], 'y': []})
        self._x = None
        self._data = None
        self._set_data(data, x=x)  # will appropriately init above 3 variables
        self.plot_xs = self._x if plot_xs is None else plot_xs  # depends on _x
        fig.circle_dot('x', 'y', source=self._data_ds, fill_color='red', line_color=None)
        # setup the model, params, and associated model parameter table
        self._model = None
        self._params = None
        self._param_pane = None
        self._model_ds = bm.ColumnDataSource({'x': [], 'y': []})
        self._comp_dss = {}  # component data sources
        self._comp_grs = {}  # component glyph renderers
        self.set_model(model, params=params)  # will appropriately init above 6 variables
        self._set_selector_options()  # now that we have the model, update the component selector
        fig.line('x', 'y', source=self._model_ds)

        # --- Setup interactive callbacks ---
        # callback to draw the polygon selector when a component is selected to edit
        self._selector_pane.link(_bokeh_pane, callbacks={'value': self.draw_selector_for_component})
        # callback to redraw the component when the polygon selector changes
        self._pdt_ds.on_change('data', self.update_component_from_selector)
        # callback to redraw the component & polygon selector when there is a scroll event
        fig.on_event(be.MouseWheel, self.update_component_params_from_scroll)
        # callback to run the fit when clicked
        self._fit_pane.on_click(self.update_component_params_from_fit)
        # callback to set the selector to None when we click away from the edit tab
        self._tabs_pane.param.watch(self.update_selector_pane_from_tab_select, ['active'])
        # callback to disable window scrolling when we are editing a component
        # noinspection PyTypeChecker
        self._selector_pane.jscallback(args={'selector': self._selector_pane}, value="""
        // need to hold on to function reference, so we can't redefine it
        if (!selector.hasOwnProperty('__hopefully_disused_name'))
           selector.__hopefully_disused_name = function(e) { e.preventDefault(); }
        
        if (selector.value == "None") {
           window.removeEventListener('wheel', selector.__hopefully_disused_name, { passive: false });
        } else {
           window.addEventListener('wheel', selector.__hopefully_disused_name, { passive: false });
        }
        """)  # alternatively, this could be done when the mouse enters/leaves the Bokeh plot area...
        # also not really sure why this doesn't suppress events to Bokeh, but it doesn't, which is nice

        self.thread = None

    def update_component_params_from_fit(self, event):
        """ Callback to fit & update our parameters due to a fit request """
        self._fit_pane.button_type = 'warning'
        self._fit_pane.name = "fit in progress..."

        result = self._model.fit(self._data, x=self._x, params=self._params)
        self._last_fit_result = result
        self._last_fit_result_time = datetime.now()

        # update fit results pane
        timeline = f"Fit Results from {self._last_fit_result_time}\n"
        self._fit_results_pane.object = timeline + self._last_fit_result.fit_report()

        self._params.update_existing_values(result.params)
        self._fit_pane.button_type = 'default'
        self._fit_pane.name = "Fit"

    def update_component_params_from_scroll(self, event):
        """ Callback to update the component when the scroll wheel is spun """
        component = self.get_selected_component()
        if component is None:
            return
        new_params = component.update_params_from_scroll(event.delta, params=self._params)
        self._params.update_existing_values(new_params)
        # redraw is triggered by the parameters updating

    def update_component_from_selector(self, attr, old, new):
        """ Callback to update the component when the polygon selector is dragged"""
        if attr != 'data':
            return
        points = dict(self._pdt_ds.data)
        component = self.get_selected_component()
        if component is None:
            return

        new_params = component.from_poly_selector_shape(points['x'][0], points['y'][0])
        # plain update will overwrite the Parameter objects, destroying our links
        self._params.update_existing_values(new_params)

    def draw_selector_for_component(self, target, event):
        """ Callback to redraw the polygon selector when a component is selected for editing """

        def subfunc():
            component = self.get_selected_component()
            if component is None:
                # we didn't find the component (so it's probably None), so don't show any points
                self._pdt_ds.data = dict(x=[[]], y=[[]])
            else:
                x, y = component.to_poly_selector_shape((self._x.min(), self._x.max()), self._params)
                self._pdt_ds.data = dict(x=[x], y=[y])

            for c, gr in self._comp_grs.items():
                gr.glyph.line_color = 'green' if c == getattr(component, 'prefix', None) else '#1F77B4'

        # since we screw with the figure directly, this needs to be wrapped
        self._fig_change(subfunc)

    def update_selector_pane_from_tab_select(self, *events):
        """ Callback to manage disabling the editing features when we switch out of the edit tab """
        # this is mostly just so the scroll wheel works when you switch tabs
        if self._tabs_pane.active == 0:
            # restores previous value
            self._selector_pane.value = self._last_selector_pane_value
        else:
            self._last_selector_pane_value = self._selector_pane.value
            self._selector_pane.value = 'None'

    def do_fit(self):
        """ Perform a fit of the current Model with current parameters

        Returns:
            a `ModelResult` object describing the fit
        """
        self.update_component_params_from_fit(None)
        return self._last_fit_result

    @property
    def panel(self):
        """ The main Panel application associated with this guess session """
        return self._panel

    @property
    def params(self):
        """ The current parameters that have been chosen """
        return self._params

    @property
    def model(self):
        """ The model current being fit """
        return self._model

    @property
    def data(self):
        """ The dependent data (y-axis) currently being fit """
        return self._data

    @property
    def residuals(self):
        """ The residuals of the current model & data being fit """
        return self._model.eval(params=self._params, x=self._x) - self._data

    @property
    def x(self):
        """ The independent data (x-axis) currently being fit """
        return self._x

    @property
    def fit(self):
        """ The `ModelResult` object from the last fit performed """
        return self._last_fit_result

    def start(self, threaded=True, **kwargs):
        """ Starts the front-end Panel application on a separate thread """
        self.thread = self._panel.show(threaded=threaded, **kwargs)

    def stop(self):
        """ Stops the front-end Panel application """
        if self.thread is not None:
            self.thread.stop()

    def is_alive(self):
        """ Returns whether the front-end Panel application is still running """
        if self.thread is not None:
            return self.thread.is_alive()

    def join(self):
        """ Blocks until the front-end application shuts down """
        if self.thread is not None:
            self.thread.join()
