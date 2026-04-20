import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections.abc import Callable
from pathlib import Path

from magie.utils import enforce_types, get_asset_path


class ArgumentError(Exception):
    """Error raised when plotting helper arguments are invalid."""


@enforce_types(ax=object, url=str)
def add_image(ax, url):
    """
    Download an image and attach it to a Matplotlib axis as an annotation artist.

    Parameters
    ----------
    ax : object
        Matplotlib axis that will receive the image artist.
    url : str
        URL of the image to download.

    Returns
    -------
    object
        Artist returned by ``ax.add_artist()``.
    """
    from io import BytesIO
    from urllib.request import urlopen
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    img_data = urlopen(url).read()
    img = Image.open(BytesIO(img_data)).convert("RGBA")
    arr = np.asarray(img)

    # create offset image (zoom controls displayed size)
    imagebox = OffsetImage(arr, zoom=0.3)
    ab = AnnotationBbox(imagebox, (0.5, 0), frameon=False)  # attach at data coords
    return ax.add_artist(ab)


@enforce_types(val=(int, float, np.ndarray, list, tuple, np.number), norm=object, cmap=(str, object))
def get_color(val, norm, cmap):
    """
    Return RGBA color values for scalar or array-like data.

    Parameters
    ----------
    val : scalar or array-like
        Input value or values to map through the colormap.
    norm : object
        Matplotlib normalization object.
    cmap : str or object
        Colormap name or Matplotlib colormap object.

    Returns
    -------
    tuple or numpy.ndarray
        RGBA tuple for scalar input or an ``(N, 4)`` array for array input.
        NaNs are converted to fully transparent values.
    """
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    arr = np.asarray(val)
    if arr.ndim == 0:
        if np.isnan(arr):
            return (0.0, 0.0, 0.0, 0.0)
        return tuple(sm.to_rgba(float(arr)))
    rgba = np.asarray(sm.to_rgba(arr))
    if np.isnan(arr).any():
        nan_mask = np.isnan(arr)
        rgba[nan_mask, :] = (0.0, 0.0, 0.0, 0.0)
    return rgba


@enforce_types(ax=object, grid=object, resolution=str, facecolor=str)
def add_land(ax, grid, resolution='50m', facecolor='darkgreen', **kwargs):
    """
    Add land polygons to a map axis using projected coastlines from a grid object.

    Parameters
    ----------
    ax : object
        Matplotlib axis that will receive the polygon patches.
    grid : object
        Grid object providing ``projection.get_projected_coastlines()``.
    resolution : str, optional
        Coastline resolution passed to the grid projection.
    facecolor : str, optional
        Fill color used for land polygons.
    **kwargs : dict
        Additional keyword arguments passed to ``PathPatch``.
    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    plot_kwargs = dict(facecolor=facecolor, edgecolor='none', zorder=1)
    plot_kwargs.update(kwargs)
    land_patches = []
    for cl in grid.projection.get_projected_coastlines(resolution=resolution):
        x, y = cl
        if len(x) < 3:
            continue  # can't form a polygon
        # Build a closed Path; assume each cl represents a coastline ring
        verts = np.column_stack([x, y])
        # ensure closed
        if not (verts[0] == verts[-1]).all():
            verts = np.vstack([verts, verts[0]])

        # Path codes: MOVETO, then LINETO..., then CLOSEPOLY
        codes = np.full(len(verts), Path.LINETO, dtype=np.uint8)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY

        path = Path(verts, codes)
        patch = PathPatch(path, **plot_kwargs)  # land under outlines/grid
        land_patches.append(patch)

    for p in land_patches:
        ax.add_patch(p)


@enforce_types(ax=object, grid=object, facecolor=str)
def add_ocean(ax, grid, facecolor='#3498db', **kwargs):
    """
    Add a rectangular ocean background patch covering the full grid extent.

    Parameters
    ----------
    ax : object
        Matplotlib axis that will receive the rectangle patch.
    grid : object
        Grid object exposing ``xi_min``, ``xi_max``, ``eta_min``, and ``eta_max``.
    facecolor : str, optional
        Fill color used for the ocean patch.
    **kwargs : dict
        Additional keyword arguments passed to ``Rectangle``.
    """
    from matplotlib.patches import Rectangle

    plot_kwargs = dict(facecolor=facecolor, edgecolor='none', zorder=-5)
    plot_kwargs.update(kwargs)

    x0, x1 = grid.xi_min, grid.xi_max
    y0, y1 = grid.eta_min, grid.eta_max

    ocean_patch = Rectangle((x0, y0), x1 - x0, y1 - y0, **plot_kwargs)
    ax.add_patch(ocean_patch)


@enforce_types(
    contour=object,
    xpad=(int, float, type(None)),
    ypad=(int, float, type(None)),
    sides=(list, str),
    rtol=(int, float),
    fmt=Callable,
    x_splits=(list, str, type(None)),
    y_splits=(list, str, type(None)),
)
def contour_labels(contour, xpad=None, ypad=None, sides=['left', 'right'], rtol=.1, fmt=lambda x: f'{x}', x_splits=None, y_splits=None, **text_kwargs):
    """
    Create contour labels positioned along the edges of a subplot.

    Parameters
    ----------
    contour : object
        Matplotlib contour set object.
    xpad : int or float or None, optional
        Horizontal offset applied to labels on the left or right edges. When
        omitted, a small fraction of the axis width is used.
    ypad : int or float or None, optional
        Vertical offset applied to labels on the top or bottom edges. When
        omitted, a small fraction of the axis height is used.
    sides : list or str, optional
        Plot edges on which labels should be created. Valid values are
        ``left``, ``right``, ``top``, and ``bottom``.
    rtol : int or float, optional
        Relative tolerance passed to ``numpy.isclose`` when detecting
        intersections with axis limits.
    fmt : collections.abc.Callable, optional
        Formatter that converts a contour level to label text.
    x_splits : list or str or None, optional
        Optional filters that restrict labels on top or bottom edges to
        positive, negative, or expression-based x regions.
    y_splits : list or str or None, optional
        Optional filters that restrict labels on left or right edges to
        positive, negative, or expression-based y regions.
    **text_kwargs : dict
        Additional keyword arguments passed to ``Axes.text``.

    Raises
    ------
    ArgumentError
        Raised in an argument is not usable.

    Returns
    -------
    list
        Created Matplotlib text artists. When multiple sides are requested, the
        result is a list of per-side text-object lists.

    """
    import numpy as np
    ax= contour.axes
    if xpad is None:
        xpad= np.max(np.abs(ax.get_xlim()))*.015
    if ypad is None:
        ypad= np.max(np.abs(ax.get_ylim()))*.015
    if isinstance(sides, str):
        sides= [sides]
    if x_splits is None or isinstance(x_splits, str):
        x_splits= [x_splits]*len(sides)
    if y_splits is None or isinstance(y_splits, str):
        y_splits= [y_splits]*len(sides)
    labels= [[] for i in range(len(sides))]
    if hasattr(contour, "collections"):
        level_paths = [
            (level, [path.vertices for path in collection.get_paths()])
            for collection, level in zip(contour.collections, contour.levels)
        ]
    else:
        level_paths = [
            (level, [seg for seg in segs])
            for level, segs in zip(contour.levels, contour.allsegs)
        ]
    for level, paths in level_paths:
        if not len(paths):
            continue
        for i, (side, x_split, y_split) in enumerate(zip(sides, x_splits, y_splits)):
            x, y= np.concatenate(paths, axis=0).T
            if side=='left':
                if y_split=='negative':
                    x= x[y<0]
                    y= y[y<0]
                elif y_split=='positive':
                    x= x[y>0]
                    y= y[y>0]
                elif not y_split is None and ('<' in y_split or '>' in y_split):
                    x= x[eval(y_split)]
                    y= y[eval(y_split)]
                elif not y_split is None:
                    raise ArgumentError(f"y_split not understood! must be either or a combination of: 'negative', 'positive' or None. Where they align with side.\n you chose {y_split}")
                if not np.any(np.isclose(x, ax.get_xlim()[0], atol=0, rtol=rtol)):
                    continue
                y= y[np.isclose(x, ax.get_xlim()[0], atol=0, rtol=rtol)]
                x= x[np.isclose(x, ax.get_xlim()[0], atol=0, rtol=rtol)]
                y= y[np.argmin(abs(x-ax.get_xlim()[0]))]
                x= x[np.argmin(abs(x-ax.get_xlim()[0]))]
                x= ax.get_xlim()[0]
                Xpad=xpad*-1
                Ypad=0
            elif side=='right':
                if y_split=='negative':
                    x= x[y<0]
                    y= y[y<0]
                elif y_split=='positive':
                    x= x[y>0]
                    y= y[y>0]
                elif not y_split is None and ('<' in y_split or '>' in y_split):
                    x= x[eval(y_split)]
                    y= y[eval(y_split)]
                elif not y_split is None:
                    raise ArgumentError(f"y_split not understood! must be either or a combination of: 'negative', 'positive' or None. Where they align with side.\n you chose {y_split}")
                if not np.any(np.isclose(x, ax.get_xlim()[1], atol=0, rtol=rtol)):
                    continue
                y= y[np.isclose(x, ax.get_xlim()[1], atol=0, rtol=rtol)]
                x= x[np.isclose(x, ax.get_xlim()[1], atol=0, rtol=rtol)]
                y= y[np.argmin(abs(x-ax.get_xlim()[1]))]
                x= x[np.argmin(abs(x-ax.get_xlim()[1]))]
                x= ax.get_xlim()[1]
                Xpad=xpad
                Ypad=0
            elif side=='bottom':
                if x_split=='negative':
                    y= y[x<0]
                    x= x[x<0]
                elif x_split=='positive':
                    y= y[x>0]
                    x= x[x>0]
                elif not x_split is None and ('<' in x_split or '>' in x_split):
                    y= y[eval(x_split)]
                    x= x[eval(x_split)]
                elif not x_split is None:
                    raise ArgumentError(f"x_split not understood! must be either or a combination of: 'negative', 'positive' or None. Where they align with side.\n you chose {x_split}")
                if not np.any(np.isclose(y, ax.get_ylim()[0], atol=0, rtol=rtol)):
                    continue
                x= x[np.isclose(y, ax.get_ylim()[0], atol=0, rtol=rtol)]
                y= y[np.isclose(y, ax.get_ylim()[0], atol=0, rtol=rtol)]
                x= x[np.argmin(abs(y-ax.get_ylim()[0]))]
                y= y[np.argmin(abs(y-ax.get_ylim()[0]))]
                y= ax.get_ylim()[0]
                Xpad=0
                Ypad=ypad*-1
            elif side=='top':
                if x_split=='negative':
                    y= y[x<0]
                    x= x[x<0]
                elif x_split=='positive':
                    y= y[x>0]
                    x= x[x>0]
                elif not x_split is None and ('<' in x_split or '>' in x_split):
                    y= y[eval(x_split)]
                    x= x[eval(x_split)]
                elif not x_split is None:
                    raise ArgumentError(f"x_split not understood! must be either or a combination of: 'negative', 'positive' or None. Where they align with side.\n you chose {x_split}")
                if not np.any(np.isclose(y, ax.get_ylim()[1], atol=0, rtol=rtol)):
                    continue
                x= x[np.isclose(y, ax.get_ylim()[1], atol=0, rtol=rtol)]
                y= y[np.isclose(y, ax.get_ylim()[1], atol=0, rtol=rtol)]
                x= x[np.argmin(abs(y-ax.get_ylim()[1]))]
                y= y[np.argmin(abs(y-ax.get_ylim()[1]))]
                y= ax.get_ylim()[1]
                Xpad=0
                Ypad=ypad
            else:
                raise ArgumentError(f"Invalid choice for side. Please choose either or any combination of: 'left', 'right', 'top' or 'bottom'\n you chose: {side}")
            labels[i].append(ax.text(x+Xpad, y+Ypad, fmt(level), va='center', ha='center', **text_kwargs))
    if len(sides)==1:
        return labels[0]
    return labels


@enforce_types(
    data=object,
    logo_path=(str, Path, type(None)),
    auto_xlim=bool,
)
def plot_BxByBz(data, logo_path=None, auto_xlim=True):
    """
    Plot the X, Y, and Z magnetic field components as stacked time series.

    Parameters
    ----------
    data : object
        Time-indexed magnetic dataset supporting ``copy()``, ``filter()``, and
        column access for ``time``, ``x``, ``y``, and ``z``.
    logo_path : str or pathlib.Path or None, optional
        Optional path to a logo image to place in the lower-right corner of
        each subplot. When omitted, the packaged MagIE logo is used.
    auto_xlim : bool, optional
        When ``True``, set each subplot x-axis to the day-bounded extent of
        the provided time series.

    Returns
    -------
    tuple
        ``(fig, ax_Bx, ax_By, ax_Bz)`` containing the created Matplotlib
        figure and axes.
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    import matplotlib.image as mpimg
    from matplotlib.ticker import MaxNLocator
    import matplotlib.dates as mdates

    data = data.copy()  # avoid mutating caller data
    data= data.filter()
    fig= plt.figure(figsize=(30, 15))
    gs= fig.add_gridspec(3, 1, hspace=0.2)
    ax_Bx= fig.add_subplot(gs[0])
    ax_By= fig.add_subplot(gs[1])
    ax_Bz= fig.add_subplot(gs[2])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if logo_path is None:
        with get_asset_path("MagIE-logo.png") as default_logo_path:
            logo = mpimg.imread(default_logo_path)
    else:
        logo = mpimg.imread(logo_path)

    logo[..., :3] = 1.0 - logo[..., :3]  # invert RGB, keep alpha
    for ax, col, color in zip([ax_Bx, ax_By, ax_Bz], ['x', 'y', 'z'], colors):
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, which='both', linestyle='--', alpha=1, lw=1)
        # --- Major ticks: days ---
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b-%Y'))

        # --- Minor ticks: hours ---
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))

        # Styling
        ax.tick_params(axis='x', which='major', labelsize=25)
        ax.tick_params(axis='y', which='major', labelsize=25)

        ax.tick_params(axis='x', which='minor', labelsize=25)

        if auto_xlim:
            ax.set_xlim(np.array(data['time']).astype('datetime64[D]').min()+np.timedelta64(1, 'D'), np.array(data['time']).astype('datetime64[D]').max()+np.timedelta64(1, 'D'))
        ax.plot(data['time'], data[col], color=color)
        ax.set_ylabel(f'B{col} [nT]', size=50)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        for date in np.unique(np.array(data['time']).astype('datetime64[D]')): ax.axvline(date, color='black')
        imagebox = OffsetImage(logo, zoom=0.1)
        ab = AnnotationBbox(
            imagebox,
            (0.98, 0.02),  # bottom-right in axes coords
            xycoords=ax.transAxes,
            frameon=False,
            box_alignment=(1, 0),
            zorder=0,
        )
        ax.add_artist(ab)

    for ax in [ax_Bx, ax_By]:
        ax.sharex(ax_Bz)
        ax.tick_params(labelbottom=False, which='both', bottom=True, top=True, left=True, right=True)
    
    return fig, ax_Bx, ax_By, ax_Bz

@enforce_types(
    data=object,
    logo_path=(str, Path, type(None)),
    auto_xlim=bool,
)
def plot_dH(data, logo_path=None, auto_xlim=True):
    """
    Plot declination, horizontal intensity, and the first difference of H.

    Parameters
    ----------
    data : object
        Time-indexed magnetic dataset supporting ``copy()``, ``filter()``, and
        column access for ``time``, ``x``, and ``y``.
    logo_path : str or pathlib.Path or None, optional
        Optional path to a logo image to place in the lower-right corner of
        each subplot. When omitted, the packaged MagIE logo is used.
    auto_xlim : bool, optional
        When ``True``, set each subplot x-axis to the day-bounded extent of
        the provided time series.

    Returns
    -------
    tuple
        ``(fig, ax_D, ax_H, ax_dH)`` containing the created Matplotlib figure
        and axes.
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    import matplotlib.image as mpimg
    from matplotlib.ticker import MaxNLocator
    import matplotlib.dates as mdates

    data = data.copy()  # avoid mutating caller data
    data= data.filter()
    fig= plt.figure(figsize=(30, 15))
    gs= fig.add_gridspec(3, 1, hspace=0.2)
    ax_D= fig.add_subplot(gs[0])
    ax_H= fig.add_subplot(gs[1])
    ax_dH= fig.add_subplot(gs[2])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    H = np.sqrt(data['x']**2 + data['y']**2)

    dHdt = np.diff(H, prepend=np.nan)

    ratio = np.divide(
        data['y'],
        H,
        out=np.full(H.shape, np.nan, dtype=float),
        where=H != 0,
    )
    D = np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0)))





    if logo_path is None:
        with get_asset_path("MagIE-logo.png") as default_logo_path:
            logo = mpimg.imread(default_logo_path)
    else:
        logo = mpimg.imread(logo_path)

    logo[..., :3] = 1.0 - logo[..., :3]  # invert RGB, keep alpha
    for ax, col, color in zip([ax_D, ax_H, ax_dH], [D, H, dHdt], colors):
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, which='both', linestyle='--', alpha=1, lw=1)
        # --- Major ticks: days ---
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b-%Y'))

        # --- Minor ticks: hours ---
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))

        # Styling
        ax.tick_params(axis='x', which='major', labelsize=25)
        ax.tick_params(axis='y', which='major', labelsize=25)

        ax.tick_params(axis='x', which='minor', labelsize=25)

        if auto_xlim:
            ax.set_xlim(np.array(data['time']).astype('datetime64[D]').min()+np.timedelta64(1, 'D'), np.array(data['time']).astype('datetime64[D]').max()+np.timedelta64(1, 'D'))
        ax.plot(data['time'], col, color=color)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        for date in np.unique(np.array(data['time']).astype('datetime64[D]')): ax.axvline(date, color='black')
        imagebox = OffsetImage(logo, zoom=0.1)
        ab = AnnotationBbox(
            imagebox,
            (0.98, 0.02),  # bottom-right in axes coords
            xycoords=ax.transAxes,
            frameon=False,
            box_alignment=(1, 0),
            zorder=0,
        )
        ax.add_artist(ab)

    for ax in [ax_D, ax_H]:
        ax.sharex(ax_dH)
        ax.tick_params(labelbottom=False, which='both', bottom=True, top=True, left=True, right=True)
    ax_D.set_ylabel('D [degress]', size=40)
    ax_H.set_ylabel('H [nT]', size=40)
    ax_dH.set_ylabel(r'$\frac{dH}{dt}$ [nT/min]', size=40)
    ymax = np.nanmax(np.abs(dHdt))
    ax_dH.set_ylim(-ymax, ymax)
    return fig, ax_D, ax_H, ax_dH
