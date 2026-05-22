import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pathlib
import datetime as dt
import pandas as pd
import matplotlib.dates as mdates
import warnings
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from collections.abc import Callable
from pathlib import Path
from typing import Any

from magie.utils import enforce_types, get_asset_path
from magie.Data_Processing import read_IAGA2002

ColorLike = str | tuple[float, ...] | list[float] | np.ndarray


@enforce_types(color=(str, tuple, list, np.ndarray))
def _relative_luminance(color: ColorLike) -> float:
    """
    Return WCAG relative luminance for any Matplotlib color.

    Parameters
    ----------
    color : str, tuple, list, or numpy.ndarray
        Color accepted by ``matplotlib.colors.to_rgb``.

    Returns
    -------
    float
        Relative luminance in the range 0 to 1.
    """
    rgb = np.asarray(mcolors.to_rgb(color))
    # Convert sRGB to linear RGB before applying the perceptual luminance weights.
    return float(
        np.where(
            rgb <= 0.03928,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4,
        )
        @ np.array([0.2126, 0.7152, 0.0722])
    )


@enforce_types(
    color=(str, tuple, list, np.ndarray),
    background=(str, tuple, list, np.ndarray),
)
def _contrast_ratio(color: ColorLike, background: ColorLike) -> float:
    """
    Return the WCAG contrast ratio between two Matplotlib colors.

    A value of 1 means no contrast; larger values are easier to distinguish.
    """
    fg_luminance = _relative_luminance(color)
    bg_luminance = _relative_luminance(background)
    lighter = max(fg_luminance, bg_luminance)
    darker = min(fg_luminance, bg_luminance)
    return (lighter + 0.05) / (darker + 0.05)


@enforce_types(
    color=(str, tuple, list, np.ndarray),
    background=(str, tuple, list, np.ndarray),
    min_contrast=(int, float),
)
def _adjust_color_for_contrast(
    color: ColorLike,
    background: ColorLike,
    min_contrast: float = 3.0,
) -> ColorLike:
    """
    Preserve hue while adjusting value until a color is readable.

    Parameters
    ----------
    color : str, tuple, list, or numpy.ndarray
        Foreground color accepted by Matplotlib.
    background : str, tuple, list, or numpy.ndarray
        Background color used for the contrast check.
    min_contrast : int or float, optional
        Minimum WCAG contrast ratio to target.

    Returns
    -------
    str, tuple, list, or numpy.ndarray
        Original color when it already meets the contrast target, otherwise a
        hex color adjusted for readability.
    """
    if _contrast_ratio(color, background) >= min_contrast:
        return color

    # HSV lets us keep the user's chosen hue/saturation and only alter brightness.
    hue, saturation, value = mcolors.rgb_to_hsv(mcolors.to_rgb(color))
    bg_luminance = _relative_luminance(background)
    # Darken colors on light backgrounds; lighten colors on dark backgrounds.
    target_value = 0.15 if bg_luminance > 0.5 else 0.95

    # Stop at the first brightness step that meets the requested contrast.
    for candidate_value in np.linspace(value, target_value, 20):
        candidate = mcolors.hsv_to_rgb((hue, saturation, candidate_value))
        if _contrast_ratio(candidate, background) >= min_contrast:
            return mcolors.to_hex(candidate)

    # Last resort for colors that cannot reach the contrast target by value alone.
    return "#595959" if bg_luminance > 0.5 else "#d9d9d9"


@enforce_types(
    background=(str, tuple, list, np.ndarray),
    n=int,
    min_contrast=(int, float),
)
def _component_line_colors(
    background: ColorLike,
    n: int = 3,
    min_contrast: float = 3.0,
) -> list[ColorLike]:
    """
    Return line colors from the active cycle adjusted for the axis background.

    Parameters
    ----------
    background : str, tuple, list, or numpy.ndarray
        Axis background color used for contrast checks.
    n : int, optional
        Number of colors to return.
    min_contrast : int or float, optional
        Minimum WCAG contrast ratio for each color.
    """
    colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    if not colors:
        colors = list(plt.rcParamsDefault["axes.prop_cycle"].by_key()["color"])
    # Repeat short cycles so callers can request any number of line colors.
    while len(colors) < n:
        colors.extend(colors)

    return [
        _adjust_color_for_contrast(color, background, min_contrast)
        for color in colors[:n]
    ]


class ArgumentError(Exception):
    """Error raised when plotting helper arguments are invalid."""


@enforce_types(ax=object, url=str)
def add_image(ax: Any, url: str) -> Any:
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
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    img_data = urlopen(url).read()
    img = Image.open(BytesIO(img_data)).convert("RGBA")
    arr = np.asarray(img)

    # create offset image (zoom controls displayed size)
    imagebox = OffsetImage(arr, zoom=0.3)
    ab = AnnotationBbox(imagebox, (0.5, 0), frameon=False)  # attach at data coords
    return ax.add_artist(ab)


@enforce_types(
    val=(int, float, np.ndarray, list, tuple, np.number),
    norm=object,
    cmap=(str, object),
)
def get_color(
    val: int | float | np.number | np.ndarray | list[Any] | tuple[Any, ...],
    norm: Any,
    cmap: str | Any,
) -> tuple[float, float, float, float] | np.ndarray:
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
def add_land(
    ax: Any,
    grid: Any,
    resolution: str = "50m",
    facecolor: str = "darkgreen",
    **kwargs: Any,
) -> None:
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
        # Build a closed path; each coastline entry is treated as a polygon ring.
        verts = np.column_stack([x, y])
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
def add_ocean(
    ax: Any,
    grid: Any,
    facecolor: str = "#3498db",
    **kwargs: Any,
) -> None:
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
def contour_labels(
    contour: Any,
    xpad: int | float | None = None,
    ypad: int | float | None = None,
    sides: list[str] | str = ["left", "right"],
    rtol: int | float = 0.1,
    fmt: Callable[[Any], str] = lambda x: f"{x}",
    x_splits: list[str | None] | str | None = None,
    y_splits: list[str | None] | str | None = None,
    **text_kwargs: Any,
) -> list[Any]:
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
        Raised if an edge or split argument is not usable.

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
        # Matplotlib has exposed contour geometry through both collections and
        # allsegs across versions; support either representation.
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
                    # Preserve the historical expression-filter interface.
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
                    # Preserve the historical expression-filter interface.
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
                    # Preserve the historical expression-filter interface.
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
                    # Preserve the historical expression-filter interface.
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
    show_logo=bool,
    auto_xlim=bool,
)
def plot_BxByBz(
    data: Any,
    logo_path: str | Path | None = None,
    show_logo: bool = False,
    auto_xlim: bool = True,
) -> tuple[Any, Any, Any, Any]:
    """
    Plot the X, Y, and Z magnetic field components as stacked time series.

    Parameters
    ----------
    data : object
        Time-indexed magnetic dataset supporting ``copy()``, ``filter()``, and
        column access for ``time``, ``x``, ``y``, and ``z``.
    logo_path : str or pathlib.Path or None, optional
        Optional path to a logo image to place in the lower-right corner of
        each subplot. When omitted, the packaged MagIE logo is used if
        ``show_logo`` is true.
    show_logo : bool, default False
        Whether to add the MagIE logo overlay to each subplot.
    auto_xlim : bool, default True
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

    # Use one shared figure with vertically stacked axes for component comparison.
    fig= plt.figure(figsize=(400/96, 378/96))
    gs= fig.add_gridspec(3, 1, hspace=0.2)
    ax_Bx= fig.add_subplot(gs[0])
    ax_By= fig.add_subplot(gs[1])
    ax_Bz= fig.add_subplot(gs[2])
    colors = _component_line_colors(ax_Bx.get_facecolor())

    if show_logo:
        # Load the logo once and reuse it on each component subplot.
        if logo_path is None:
            with get_asset_path("MagIE-logo.png") as default_logo_path:
                logo = mpimg.imread(default_logo_path)
        else:
            logo = mpimg.imread(logo_path)

        logo[..., :3] = 1.0 - logo[..., :3]  # invert RGB, keep alpha
    for ax, col, color in zip([ax_Bx, ax_By, ax_Bz], ['x', 'y', 'z'], colors):
        ax.tick_params(axis='both', which='major')
        # Reset grid state before applying separate x/y grid styles.
        ax.grid(False)

        # Vertical grid lines mark six-hour minor ticks.
        ax.grid(True, which='minor', axis='x', linestyle='--', alpha=1, lw=1)

        # Horizontal grid lines track the component scale.
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=1, lw=1)
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))

        ax.xaxis.set_major_locator(
            mdates.HourLocator(byhour=[11])  # noon
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter('%d-%b-%Y')
        )

        if auto_xlim:
            ax.set_xlim(np.array(data['time']).astype('datetime64[D]').min()+np.timedelta64(1, 'D'), np.array(data['time']).astype('datetime64[D]').max()+np.timedelta64(1, 'D'))
        ax.plot(data['time'], data[col], color=color)
        ax.set_ylabel(f'B{col} (nT)')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        # Emphasize day boundaries over the six-hour minor grid.
        for date in np.unique(np.array(data['time']).astype('datetime64[D]')): ax.axvline(date, color='black')
        if show_logo:
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

    # Hide repeated date labels while retaining aligned ticks across panels.
    for ax in [ax_Bx, ax_By]:
        ax.sharex(ax_Bz)
        ax.tick_params(labelbottom=False, which='both', bottom=True, top=True, left=True, right=True)

    # Show hour labels only on the bottom panel.
    ax_Bz.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax_Bz.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
    ax_Bz.tick_params(axis='x', which='major', labelrotation=0, pad=15)
    ax_Bz.tick_params(axis='x', which='minor', labelrotation=0, pad=2)

    return fig, ax_Bx, ax_By, ax_Bz

@enforce_types(
    data=object,
    logo_path=(str, Path, type(None)),
    show_logo=bool,
    auto_xlim=bool,
)
def plot_dH(
    data: Any,
    logo_path: str | Path | None = None,
    show_logo: bool = False,
    auto_xlim: bool = True,
) -> tuple[Any, Any, Any, Any]:
    """
    Plot declination, horizontal intensity, and the first difference of H.

    Parameters
    ----------
    data : object
        Time-indexed magnetic dataset supporting ``copy()``, ``filter()``, and
        column access for ``time``, ``x``, and ``y``.
    logo_path : str or pathlib.Path or None, optional
        Optional path to a logo image to place in the lower-right corner of
        each subplot. When omitted, the packaged MagIE logo is used if
        ``show_logo`` is true.
    show_logo : bool, default False
        Whether to add the MagIE logo overlay to each subplot.
    auto_xlim : bool, default True
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

    # Use one shared figure with vertically stacked axes for derived quantities.
    fig= plt.figure(figsize=(400/96, 378/96))
    gs= fig.add_gridspec(3, 1, hspace=0.2)
    ax_D= fig.add_subplot(gs[0])
    ax_H= fig.add_subplot(gs[1])
    ax_dH= fig.add_subplot(gs[2])
    colors = _component_line_colors(ax_D.get_facecolor())
    H = np.sqrt(data['x']**2 + data['y']**2)

    # First difference of H at the native cadence; the label assumes minute data.
    dHdt = np.diff(H, prepend=np.nan)

    # Declination uses a clipped ratio to avoid invalid arcsin values from
    # floating-point noise or zero horizontal field strength.
    ratio = np.divide(
        data['y'],
        H,
        out=np.full(H.shape, np.nan, dtype=float),
        where=H != 0,
    )
    D = np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0)))

    if show_logo:
        # Load the logo once and reuse it on each derived-data subplot.
        if logo_path is None:
            with get_asset_path("MagIE-logo.png") as default_logo_path:
                logo = mpimg.imread(default_logo_path)
        else:
            logo = mpimg.imread(logo_path)

        logo[..., :3] = 1.0 - logo[..., :3]  # invert RGB, keep alpha
    for ax, col, color in zip([ax_D, ax_H, ax_dH], [D, H, dHdt], colors):
        ax.tick_params(axis='both', which='major')
        # Reset grid state before applying separate x/y grid styles.
        ax.grid(False)

        # Vertical grid lines mark six-hour minor ticks.
        ax.grid(True, which='minor', axis='x', linestyle='--', alpha=1, lw=1)

        # Horizontal grid lines track each panel's y-axis scale.
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=1, lw=1)

        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))

        ax.xaxis.set_major_locator(
            mdates.HourLocator(byhour=[11])  # noon
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter('%d-%b-%Y')
        )


        if auto_xlim:
            ax.set_xlim(np.array(data['time']).astype('datetime64[D]').min()+np.timedelta64(1, 'D'), np.array(data['time']).astype('datetime64[D]').max()+np.timedelta64(1, 'D'))
        ax.plot(data['time'], col, color=color)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        # Emphasize day boundaries over the six-hour minor grid.
        for date in np.unique(np.array(data['time']).astype('datetime64[D]')): ax.axvline(date, color='black')
        if show_logo:
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

    # Hide repeated date labels while retaining aligned ticks across panels.
    for ax in [ax_D, ax_H]:
        ax.sharex(ax_dH)
        ax.tick_params(labelbottom=False, which='both', bottom=True, top=True, left=True, right=True)

    # Show hour labels only on the bottom panel.
    ax_dH.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax_dH.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))


    ax_dH.tick_params(axis='x', which='major', labelrotation=0, pad=15)
    ax_dH.tick_params(axis='x', which='minor', labelrotation=0, pad=2)
    ax_D.set_ylabel('D (degrees)')
    ax_H.set_ylabel('H (nT)')
    ax_dH.set_ylabel(r'$\frac{dH}{dt}$ (nT/min)')
    ymax = np.nanmax(np.abs(dHdt))
    # Keep positive and negative dH excursions visually comparable.
    ax_dH.set_ylim(-ymax, ymax)
    return fig, ax_D, ax_H, ax_dH


@enforce_types(
    df=pd.DataFrame,
    obs_plot_list=list,
    padding_fraction=float,
    component_list=list,
    means=dict,
    scale_length=(int, float),
    y_labels=list,
    title=str,
    file_name=str,
    output_file_path=pathlib.Path,
    font_size=int,
    title_font=int,
    print_msg=bool,
)
def stack_plot(df, obs_plot_list, padding_fraction, component_list,
               means, scale_length, y_labels, title, file_name,
               output_file_path, font_size=15, title_font=18,
               print_msg=False):
    """
    --- Create 3-panel subplot of observatory timeseries ---
    A station with a naturally large X, Y and Z values
    ends up far away from the others. To deal with this we must:
    Step 1. subtract the absolute baseline - treated as mean_xyz[obs]
    in each component so all stations are near zero.
    Step 2. Apply equal vertical offsets
    y_adjusted = y_original - y_baseline + offset
    Draws a 500 nT scale bar at the top right corner of each panel.
    Author: Guanren Wang (gwang1@tcd.ie)

    Parameters:
    -----------
    df: pd.DataFrame
        index timestamped as datetime object containing data with column header
        as OBSX, OBSY, OBSZ, etc

    obs_plot_list: list
        iaga three-letter observatory code in a comma separated list

    padding_fraction: float

        Used in equal-spacing offset for stacking based on maximum variation.
        Usually between 0.1 and 0.3 or 10-30% padding

    component_list: list
        strings of components in comma sepatated list E.g. ['X', 'Y, 'Z']

    means: dict
        dictionary of numpy.float values for mean in different components

    scale_length: float, int
        Scale bar in nT

    y_labels: list
        strings for y-axis label in a comma separated list

    title: str
        plot title, can be '' if no title

    file_name: str
        name of file e.g. Mid-Latitude_Nov11-12_Bxyz.png

    output_file_path: pathlib.Path

    font_size: int

    title_font: int

    print_msg: bool
        Turn off or on print messages. Default is off.

    Dependencies:
    -------------
    Function: means_calc
    Ensure your data frame index is datetime:
    df.index = pd.to_datetime(df.index)

    Returns:
    --------
    fig : matplotlib.figure.Figure
    ax : list of matplotlib.axes.Axes
    y_offsets_per_obs: list
    """
    # Define colour pallette
    N = len(obs_plot_list)
    if N < 5:
        colours = ["black", "crimson", "blue", "green"]
    else:
        colours = plt.colormaps["viridis"](np.linspace(0, 0.85, N))
    # Assign offset values per-observatory dynamically based on data range
    data_ranges = []
    for obs in obs_plot_list:
        obs_spreads = []
        for comp in component_list:
            col = f"{obs.upper()}{comp}"
            spread = df[col].max() - df[col].min()
            obs_spreads.append(spread)
        data_ranges.append(max(obs_spreads))
    # compute equal spacing offset for stacking, based on maximum variation
    equal_spacing = max(data_ranges) * padding_fraction
    if print_msg:
        print(f"Using vertical spacing of {equal_spacing:.0f} nT")
    y_offsets_per_obs = []
    # compute range per component
    for obs in obs_plot_list:
        obs = obs.upper()
        ranges_per_component = []
        for comp in ["X", "Y", "Z"]:
            comp = f"{obs}{comp}"
            obs_range = df[comp].max() - df[comp].min()
            ranges_per_component.append(obs_range)
        # use maximum range per component for spacing
        y_offsets_per_obs.append(max(ranges_per_component))
    rows = len(component_list)
    fig, axes = plt.subplots(
        nrows=rows, ncols=1, figsize=(14, 10), sharex=True
        )
    for i, ax in enumerate(axes):
        # set x-axis limits
        ax.set_xlim(df.index[0], df.index[-1])
        # Prepare lists for y-ticks and labels
        ytick_positions = []
        ytick_labels = []
        for obs_index, obs in enumerate(obs_plot_list):
            colour = colours[obs_index]
            obs = obs.upper()
            comp = f"{obs}{component_list[i]}"
            # offset values by-observatory-per-component
            # first observatory in the list gets highest offset
            y_offset = (N - 1 - obs_index) * equal_spacing
            if comp and comp.endswith(("X", "Y", "Z", "H"))\
                    and len(comp) == len(obs) + 1:
                # Add mean labels at start of each time series trace
                mean_obs = means[obs][component_list[i]]
                if print_msg:
                    print(obs, component_list[i], "mean is: ", mean_obs, "nT")
                # normalise each observatory by removing its mean
                y_val_normalised = df[comp] - mean_obs
                y_val_series = y_val_normalised + y_offset
                # label mean values in each component
                ax.text(
                    df.index[50],
                    y_offset,
                    f'{mean_obs:.0f} nT',
                    va='bottom', ha='left', fontsize=font_size
                    )
                first_y_val_normalised = df[comp].iloc[0] - mean_obs
                first_y_val = first_y_val_normalised + y_offset
            else:
                y_val_series = df[comp] + y_offset
                first_y_val = y_offset
            # plot time series
            ax.plot(
                df.index,
                y_val_series,
                label=obs, linewidth=1.2, color=colour
                )
            # y-tick location per observatory
            ytick_positions.append(first_y_val)
            ytick_labels.append(obs)
        # Set and label y-ticks after plotting all observatory time series
        ax.yaxis.set_ticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize=font_size)
        # Hide right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # x-axis formatters (will be applied only on bottom panel,
        # but safe to set for all)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 12, 24)))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d %H:%M'))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
        # Draw vertical dashed lines at major x-ticks
        for tick in ax.get_xticks():
            ax.axvline(
                x=tick, color='black', linestyle='--',
                alpha=0.5, linewidth=1.5
                )
        # Tick parameters
        ax.tick_params(
            axis='x', which='minor', direction='in',
            labelsize=15, length=5
            )
        ax.tick_params(
            axis='x', which='major', direction='out',
            labelsize=16, length=20
            )
        # --- Draw vertical scale bar in nT on the top-right of each panel ---
        ymin, ymax = ax.get_ylim()
        rel_height = scale_length / (ymax - ymin)
        x_pos, y_top = 0.95, 0.98
        ax.plot(
            [x_pos, x_pos], [y_top-rel_height, y_top],
            transform=ax.transAxes, color='black', lw=2, clip_on=False
            )
        ax.text(
            x_pos+0.01, y_top-rel_height/2, f'{scale_length} nT',
            transform=ax.transAxes,
            va='center', ha='left', fontsize=font_size, color='black'
            )
        # set y-axis label
        ax.set_ylabel(y_labels[i], fontsize=font_size)
    # Bottom panel gets the x-axis label
    axes[-1].set_xlabel('Time UTC', fontsize=font_size)
    # set title
    plt.suptitle(title, fontsize=title_font)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    # save file to output_file_path directory path
    ff = output_file_path / file_name
    if print_msg:
        print(f"Plot {file_name} is saved in {output_file_path}")
    fig.savefig(ff, format="png")
    return fig, axes, y_offsets_per_obs


@enforce_types(
    start_time=dt.datetime,
    end_time=dt.datetime,
    obs=str,
    base_dir=pathlib.Path,
    plot_title=str,
    outfile_name=str,
    footer=str,
    comps=(list, tuple),
    print_msg=bool,
    print_debug=bool,
)
def plot_variometer_data(start_time, end_time, obs, base_dir,
                         plot_title, outfile_name,
                         footer='', comps=["X", "Y", "Z"],
                         print_msg=False, print_debug=False):
    """
    Program to plot timeseries of raw variometer data for up to four components
    Handles daily file path logic.
    Author: Guanren Wang (gwang1@tcd.ie)

    Parameters:
    -----------
    start_time: datetime.datetime
        datetime.datetime(2026, 4, 29, 0, 0) or
        pd.Timestamp("2026-04-29 00:00:00")

    end_time: datetime.datetime
        datetime.datetime(2026, 5, 1, 23, 59, 59) or
        pd.Timestamp("2026-05-01 23:59:59")

    obs: str
        iaga three-letter observatory code

    base_dir: pathlib.Path
        Base folder where daily iaga-2002 day files live.
        Presently we assume directory structure is in the form of:
        base_dir/year/mon/dd/txt/

    plot_title: str
        Title to appear at the top of the plot

    outfile_name: str
        File name for figure

    footer: str
        Attrribute information E.g. "For research use only."

    comps: list or tuple
        specify components to plot defaults to ['X','Y','Z']
        E.g. ['X'], ['X','Y'], ['X','Y','Z'] or ['X','Y','Z', 'F']

    print_msg: bool
        Turn off or on print messages. Default is off

    print_debug: bool
        Turn off or on debugging messages. Default is off

    Dependencies:
    -------
    Function: plot_xyzf

    Raises:
    -------
    FileNotFoundError

    Returns:
    --------
    Saves figure as png file in base_dir/year/mon/dd/png/
    fig : matplotlib.figure.Figure
    ax : list of matplotlib.axes.Axes
    """
    df_list = []
    date_range = pd.date_range(start_time, end_time, freq="D")
    for day in date_range:
        date_str = day.strftime("%Y%m%d")
        year = day.strftime("%Y")
        mon = day.strftime("%m")
        dd = day.strftime("%d")
        # txt directory and standard txt file name
        daily_dir = base_dir / year / mon / dd / "txt"
        fname = daily_dir / f"{obs}{date_str}.txt"
        # IAGA2002 daily file directory
        iaga2002_daily_dir = base_dir / year / mon / dd / "iaga2002"
        # daily figure directory
        daily_dir_fig = base_dir / year / mon / dd / "png"
        # create daily figure directory if it does not exist already
        daily_dir_fig.mkdir(parents=True, exist_ok=True)
        try:
            if fname.exists():
                df = pd.read_csv(
                    fname, sep=r"\s+",
                    names=["Date", "Time", "Index#", "Bx", "By", "Bz"],
                    skiprows=1,
                    na_values=99999.00
                    )
                df["Date & Time"] = df["Date"] + " " + df["Time"]
                df = df.drop(columns=["Date", "Time"])
                df = df.set_index("Date & Time")
                df.index = pd.to_datetime(df.index)
                df_list.append(df)
            else:
                priority = [f"{obs.lower()}*qsec.sec",
                            f"{obs.lower()}*psec.sec",
                            f"{obs.lower()}*vsec.sec",
                            f"{obs.lower()}*dmin.min",
                            f"{obs.lower()}*qmin.min",
                            f"{obs.lower()}*pmin.min",
                            f"{obs.lower()}*vmin.min"]
                iaga_file = None
                for pattern in priority:
                    match = list(iaga2002_daily_dir.glob(pattern))
                    if match:
                        iaga_file = match[0]
                        break
                if iaga_file is None:
                    print(f"No TXT or IAGA-2002 files found in {date_str}")
                    continue
                df = read_IAGA2002(
                    iaga_file.parent, iaga_file.name, print_debug=print_debug
                    )
                df_list.append(df)
        except Exception as e:
            print(f"Error reading data for {date_str}: {e}")
    # print this if all files are missing
    if not df_list:
        raise FileNotFoundError(
            "No valid daily variometer files found between "
            f"{start_time} and {end_time}"
        )
    combined_df = pd.concat(df_list)
    rename_map = {}
    obs = obs.upper()
    for old, new in zip(
        ["Bx", "By", "Bz", "Bf"], [f"{obs}{x}" for x in ["X", "Y", "Z", "F"]]
    ):
        if old in combined_df.columns:
            rename_map[old] = new
    combined_df.rename(columns=rename_map, inplace=True)
    combined_df.drop(columns=["Index#", "DOY"], inplace=True, errors="ignore")
    # File saved in the latest daily file directory path
    ff = daily_dir_fig / outfile_name
    fig, ax = plot_xyzf(
        combined_df, obs, start_time, end_time, plot_title, comps, footer
        )
    # save file as png
    if print_msg is True:
        print(f"Saving figure {outfile_name} into {daily_dir_fig}")
        print(f"Directory path to figure is : {ff}")
    fig.savefig(ff, format="png")
    return fig, ax


@enforce_types(
    df=pd.DataFrame,
    obs=str,
    start_time=dt.datetime,
    end_time=dt.datetime,
    plot_title=str,
    comps=(list, tuple),
    footer=str,
    font_size=int,
    title_font=int
)
def plot_xyzf(df, obs, start_time, end_time, plot_title, comps,
              footer='', font_size=15, title_font=18):
    """
    Quick plot to zoom into specific time interval for one component.
    Author: Guanren Wang (gwang1@tcd.ie)

    Parameters:
    -----------
    df: pd.DataFrame
        data frame sorted and ensure it is in datetime

    obs: str
        iaga three-letter observatory code

    start_time: datetime.datetime
        datetime.datetime(2026, 4, 29, 0, 0) or
        pd.Timestamp("2026-04-29 00:00:00")

    end_time: datetime.datetime
        datetime.datetime(2026, 5, 1, 23, 59, 59) or
        pd.Timestamp("2026-05-01 23:59:59")

    plot_title: str
        E.g. f"{ob} raw one-second fluxgate data"

    comps: list or tuple
        Components to plot
        E.g. ['X'], ['X','Y'], ['X','Y','Z'] or ['X','Y','Z', 'F']

    footer: str
        Attrribute information E.g. "For research use only."

    font_size: int

    Raises:
    -------
    ValueError
    KeyError

    Returns:
    --------
    fig: matplotlib.figure.Figure
    ax: list of matplotlib.axes.Axes
    """
    df_window = df.loc[start_time:end_time]
    valid_comps = {'X', 'Y', 'Z', 'F'}
    # capitalise elements in comps if they aren't capitalised
    comps = [x.upper() for x in comps]

    invalid = [x for x in comps if x not in valid_comps]
    if invalid:
        raise ValueError(
            f"invalid component(s): {invalid}"
            f"Valid options are {sorted(valid_comps)}."
        )
    # build column names, check if requested components in comps exits in df
    cols = [f"{obs.upper()}{c}"for c in comps]
    existing_cols = [c for c in cols if c in df.columns]
    missing_cols = [c for c in cols if c not in df.columns]
    if not existing_cols:
        raise KeyError(
            f"None of the columns in comps exists: '{cols}'. "
            f"Available columns: {list(df.columns)}"
        )
    if missing_cols:
        warnings.warn(
            "The following requested columns are missing and will be skipped:"
            f"{missing_cols}"
        )
    n = len(comps)
    fig, ax = plt.subplots(
        nrows=n, ncols=1, sharex=True, figsize=(14, 3.5*n)
        )
    if n == 1:
        ax = [ax]
    # Plot using the window
    for axis, col, comp in zip(ax, cols, comps):
        axis.plot(
            df_window.index, df_window[col], color='k', label=comp, linewidth=1
            )
        # Force exactly 4 y-axis ticks
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
        subscript = comp.lower() if comp != 'F' else "F"
        axis.set_ylabel(rf"$B_{subscript}$ (nT)", fontsize=font_size)
        # y-axis tick labels (numbers along the axis)
        axis.tick_params(
            axis='y', which='major', direction='in', labelsize=15, length=10
            )
    ax[-1].set_xlim(df_window.index[0], df_window.index[-1])
    # X-axis ticks
    ax[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 12)))
    ax[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d %H:%M'))
    ax[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    ax[-1].xaxis.set_major_locator(mdates.DayLocator())
    for axis in ax:
        axis.grid(
            True, which="major", axis="x", linestyle="-",
            alpha=1, linewidth=1.5
            )
        axis.grid(
            True, which="minor", axis="both", linestyle="-",
            alpha=0.5, linewidth=1.5
            )
    # X-tick parameters
    ax[-1].tick_params(
        axis='x', which='minor', direction='in', labelsize=15, length=10
        )
    ax[-1].tick_params(
        axis='x', which='major', direction='out', labelsize=15, length=20
        )
    ax[-1].set_xlabel("Time (UT)", fontsize=font_size)
    # set title
    ax[0].set_title(plot_title, fontsize=title_font)
    if footer:
        fig.text(
            0.08, 0.01, footer,
            ha='left', va='bottom',
            fontsize=font_size, style='italic', color='black'
        )
    # leave space at bottom of footer
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig, ax
