"""Print-size figure style for an IEEE two-column page.

Every figure is drawn at the width it is printed at -- one column (COL_W) or
the full text width (PAGE_W) -- so the font sizes set here are the sizes that
appear on paper. Drawing at 6 in and letting LaTeX shrink the result into a
3.5 in column prints 8 pt labels at ~4.5 pt, which is what this replaces.

Figures carry no titles: the caption names the figure, and a title that fits
a 3.5 in column is too short to say anything the caption does not.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_W = 3.5        # single IEEE column, inches
PAGE_W = 7.16      # full IEEE text width, inches
DPI = 300          # print resolution for raster figures

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "legend.framealpha": 0.95,
    "legend.borderaxespad": 0.4,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.5,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
