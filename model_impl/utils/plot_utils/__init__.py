"""
Figure modules, split by family: training curves, forecasts, calibration and
faithfulness. Nothing here computes a metric — callers pass in already-scored
arrays and these functions only draw and save them.

The headless backend is pinned here once; importing any submodule runs this
package first, so no submodule needs its own matplotlib.use call.
"""

import matplotlib

matplotlib.use("Agg")
