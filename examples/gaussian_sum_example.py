import numpy as np
from ivfit.models import IGaussianModel

# construct data - 3 gaussian peaks with noise, against a broad gaussian background
x = np.linspace(-4, 4, num=75)
y = np.exp(-x ** 2)
x = np.linspace(-5, 25, num=75 * 3)
y = np.concatenate([y, 2 * y, 3 * y])
y += (np.random.random(len(x)) - 0.5) / 5
y += np.exp(-np.linspace(-1, 3, num=75 * 3) ** 2)

# construct model that is sum of 4 Gaussians
cm = IGaussianModel(prefix="g0_") + IGaussianModel(prefix="g1_") + \
     IGaussianModel(prefix="g2_") + IGaussianModel(prefix="bk_")

# launch interactive guess session with this data & model
igs = cm.interactive_guess(y, x)
