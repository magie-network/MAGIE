#!/usr/bin/env python3
"""Create a Plotly cubed-sphere status map for MAGIE magnetometers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from secsy.cubedsphere import CSgrid, CSprojection
from secsy.natural_earth import project_natural_earth

from magie.utils import enforce_types, get_site_metadata


DEFAULT_ACTIVE_COLOR = "#1f9d55"
DEFAULT_INACTIVE_COLOR = "#d64545"
DEFAULT_PERMANENTLY_OFF_COLOR = "#80868b"
DEFAULT_WATER_COLOR = "#dcecf4"


@enforce_types(value=(str, type(None)))
def parse_time(value: str | None) -> datetime | None:
    """
    Parse an ISO datetime string, accepting trailing ``Z`` for UTC.
    """

    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@enforce_types(start=(datetime, type(None)), end=datetime)
def human_delta(start: datetime | None, end: datetime) -> str:
    """
    Format the elapsed time between two datetimes for hover text.
    """

    if start is None:
        return "unknown"

    seconds = max(0, int((end - start).total_seconds()))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if days:
        return f"{days} d {hours} h"
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {seconds} s"
    return f"{seconds} s"


@enforce_types(value=(str, type(None)))
def display_time_without_timezone(value: str | None) -> str | None:
    """
    Format an ISO datetime string without timezone information.
    """

    parsed = parse_time(value)
    if parsed is None:
        return None
    return parsed.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


@enforce_types(path=(str, Path))
def load_status(path: str | Path) -> dict:
    """
    Load a magnetometer monitor status JSON file.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@enforce_types(
    status_data=dict,
    active_color=str,
    inactive_color=str,
    permanently_off_color=str,
)
def station_rows(
    status_data: dict,
    active_color: str = DEFAULT_ACTIVE_COLOR,
    inactive_color: str = DEFAULT_INACTIVE_COLOR,
    permanently_off_color: str = DEFAULT_PERMANENTLY_OFF_COLOR,
) -> list[dict]:
    """
    Build Plotly-ready station rows from monitor status data.
    """

    generated_at = parse_time(status_data.get("generated_at"))
    reference_time = generated_at or datetime.now(timezone.utc)
    rows = []

    for code, station_status in sorted(status_data.get("stations", {}).items()):
        metadata = get_site_metadata(code, longitude_style="signed")
        if metadata is None:
            continue

        last_measurement = parse_time(station_status.get("last_measurement"))
        status = str(station_status.get("status", "unknown"))
        status_key = status.lower()
        is_active = status_key == "online" and not station_status.get("offline", False)
        if status_key == "permanently_off":
            state = "permanently_off"
            color = permanently_off_color
        elif is_active:
            state = "active"
            color = active_color
        else:
            state = "inactive"
            color = inactive_color

        rows.append(
            {
                "code": code.upper(),
                "name": metadata.get("station_name", code.upper()),
                "lon": float(metadata["longitude"]),
                "lat": float(metadata["latitude"]),
                "status": status,
                "state": state,
                "color": color,
                "last_measurement": last_measurement,
                "last_measurement_text": (
                    last_measurement.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
                    if last_measurement
                    else "unknown"
                ),
                "time_since": human_delta(last_measurement, reference_time),
            }
        )

    return rows


@enforce_types(parts=list)
def joined_xy(parts: list[dict]) -> tuple[list[float | None], list[float | None]]:
    """
    Join projected Natural Earth geometry parts with ``None`` separators.
    """

    x = []
    y = []
    for part in parts:
        x.extend(part["x"].tolist())
        y.extend(part["y"].tolist())
        x.append(None)
        y.append(None)
    return x, y


@enforce_types()
def make_grid() -> CSgrid:
    """
    Create the cubed-sphere grid used for Ireland and nearby stations.
    """

    projection = CSprojection(position=(-8.0, 53.4), orientation=0)
    return CSgrid(projection, L=400, W=500, Lres=10.0, Wres=10.0)


@enforce_types(
    fig=go.Figure,
    grid=CSgrid,
    resolution=str,
    water_color=str,
    add_borders=bool,
)
def add_feature_traces(
    fig: go.Figure,
    grid: CSgrid,
    resolution: str,
    water_color: str,
    add_borders: bool = False,
) -> None:
    """
    Add land, water, coastline, and border traces to a Plotly figure.
    """

    land_parts = project_natural_earth(
        grid,
        "land",
        resolution=resolution,
        part_types={"polygon_exterior"},
        require_closed=True,
        geographic_buffer=2,
    )
    for part in land_parts:
        fig.add_trace(
            go.Scatter(
                x=part["x"],
                y=part["y"],
                mode="lines",
                fill="toself",
                line=dict(width=0),
                fillcolor="#edf3df",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    lake_parts = project_natural_earth(
        grid,
        "lakes",
        resolution=resolution,
        part_types={"polygon_exterior"},
        require_closed=True,
        geographic_buffer=2,
    )
    for part in lake_parts:
        fig.add_trace(
            go.Scatter(
                x=part["x"],
                y=part["y"],
                mode="lines",
                fill="toself",
                line=dict(width=0),
                fillcolor=water_color,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    line_layers = [("coastline", "#45505c", 1.4)]
    if add_borders:
        line_layers.append(("borders", "#7b8794", 0.9))

    for layer, color, width in line_layers:
        x, y = joined_xy(
            project_natural_earth(
                grid,
                layer,
                resolution=resolution,
                geographic_buffer=2,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip",
                showlegend=False,
            )
        )


@enforce_types(
    fig=go.Figure,
    grid=CSgrid,
    rows=list,
    active_color=str,
    inactive_color=str,
    permanently_off_color=str,
)
def add_station_traces(
    fig: go.Figure,
    grid: CSgrid,
    rows: list[dict],
    active_color: str = DEFAULT_ACTIVE_COLOR,
    inactive_color: str = DEFAULT_INACTIVE_COLOR,
    permanently_off_color: str = DEFAULT_PERMANENTLY_OFF_COLOR,
) -> None:
    """
    Add grouped station marker traces and fixed legend entries to a Plotly figure.
    """

    for state, label, color in (
        ("active", "Active", active_color),
        ("inactive", "Inactive", inactive_color),
        ("permanently_off", "Permanently off", permanently_off_color),
    ):
        group = [row for row in rows if row["state"] == state]
        if not group:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        size=18,
                        color=color,
                        line=dict(width=2, color="white"),
                    ),
                    name=label,
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
            continue
        xi, eta = grid.projection.geo2cube(
            np.array([row["lon"] for row in group]),
            np.array([row["lat"] for row in group]),
        )
        fig.add_trace(
            go.Scatter(
                x=xi,
                y=eta,
                mode="markers+text",
                text=[row["code"] for row in group],
                textposition="top center",
                marker=dict(
                    size=18,
                    color=[row["color"] for row in group],
                    line=dict(width=2, color="white"),
                ),
                name=label,
                customdata=[
                    [
                        row["name"],
                        row["status"],
                        row["time_since"],
                        row["last_measurement_text"],
                    ]
                    for row in group
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Status: %{customdata[1]}<br>"
                    "Time since last measurement: %{customdata[2]}<br>"
                    "Last measurement: %{customdata[3]}<extra></extra>"
                ),
            )
        )


@enforce_types(
    rows=list,
    generated_at=(str, type(None)),
    resolution=str,
    water_color=str,
    add_borders=bool,
    active_color=str,
    inactive_color=str,
    permanently_off_color=str,
)
def build_figure(
    rows: list[dict],
    generated_at: str | None,
    resolution: str,
    water_color: str = DEFAULT_WATER_COLOR,
    add_borders: bool = False,
    active_color: str = DEFAULT_ACTIVE_COLOR,
    inactive_color: str = DEFAULT_INACTIVE_COLOR,
    permanently_off_color: str = DEFAULT_PERMANENTLY_OFF_COLOR,
) -> go.Figure:
    """
    Build the complete magnetometer status map figure.
    """

    grid = make_grid()
    fig = go.Figure()
    add_feature_traces(fig, grid, resolution, water_color, add_borders=add_borders)
    add_station_traces(
        fig,
        grid,
        rows,
        active_color=active_color,
        inactive_color=inactive_color,
        permanently_off_color=permanently_off_color,
    )

    generated_at_display = display_time_without_timezone(generated_at)
    subtitle = (
        f"Generated at {generated_at_display}"
        if generated_at_display
        else "Generated time unknown"
    )
    fig.update_layout(
        title=dict(
            text=f"MagIE Magnetometer Status<br><sup>{subtitle}</sup>",
            x=0.5,
            xanchor="center",
        ),
        width=980,
        height=850,
        paper_bgcolor="#f8faf7",
        plot_bgcolor=water_color,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=0.02,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=28, r=28, t=88, b=28),
    )
    fig.update_xaxes(
        visible=False,
        range=[grid.xi_min, grid.xi_max],
        scaleanchor="y",
        scaleratio=1,
    )
    fig.update_yaxes(visible=False, range=[grid.eta_min, grid.eta_max])
    return fig


@enforce_types(
    status_json=(str, Path),
    output=(str, Path),
    natural_earth_resolution=str,
    active_color=str,
    inactive_color=str,
    permanently_off_color=str,
    water_color=str,
    add_borders=bool,
)
def create_status_map(
    status_json: str | Path,
    output: str | Path,
    natural_earth_resolution: str = "10m",
    active_color: str = DEFAULT_ACTIVE_COLOR,
    inactive_color: str = DEFAULT_INACTIVE_COLOR,
    permanently_off_color: str = DEFAULT_PERMANENTLY_OFF_COLOR,
    water_color: str = DEFAULT_WATER_COLOR,
    add_borders: bool = False,
) -> Path:
    """
    Create a magnetometer status map HTML file.

    Parameters
    ----------
    status_json : str or pathlib.Path
        Path to a monitor status JSON file.
    output : str or pathlib.Path
        Path where the HTML map should be written.
    natural_earth_resolution : {"110m", "50m", "10m"}, optional
        Natural Earth feature resolution passed to secsy.
    active_color : str, optional
        Marker colour for stations currently online.
    inactive_color : str, optional
        Marker colour for stations currently offline.
    permanently_off_color : str, optional
        Marker colour for intentionally disabled stations.
    water_color : str, optional
        Fill colour for lakes and the map background.
    add_borders : bool, optional
        Whether to draw country borders on the map.

    Returns
    -------
    pathlib.Path
        Resolved path to the written HTML file.

    Raises
    ------
    RuntimeError
        If no stations in the status data match MAGIE site metadata.
    """

    status_path = Path(status_json)
    output_path = Path(output)
    status_data = load_status(status_path)
    rows = station_rows(
        status_data,
        active_color=active_color,
        inactive_color=inactive_color,
        permanently_off_color=permanently_off_color,
    )
    if not rows:
        raise RuntimeError("No stations could be matched to MAGIE site metadata.")

    fig = build_figure(
        rows,
        generated_at=status_data.get("generated_at"),
        resolution=natural_earth_resolution,
        water_color=water_color,
        add_borders=add_borders,
        active_color=active_color,
        inactive_color=inactive_color,
        permanently_off_color=permanently_off_color,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    return output_path.resolve()


@enforce_types()
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the status map script.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "status_json",
        type=Path,
        help="Path to magnetometer_monitor_status.json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--natural-earth-resolution",
        choices=("110m", "50m", "10m"),
        default="10m",
    )
    parser.add_argument("--add-borders", action="store_true")
    parser.add_argument("--active-color", default=DEFAULT_ACTIVE_COLOR)
    parser.add_argument("--inactive-color", default=DEFAULT_INACTIVE_COLOR)
    parser.add_argument(
        "--permanently-off-color",
        default=DEFAULT_PERMANENTLY_OFF_COLOR,
    )
    parser.add_argument("--water-color", default=DEFAULT_WATER_COLOR)
    args, _unknown = parser.parse_known_args()
    return args


@enforce_types()
def main() -> None:
    """
    Run the command-line status map writer.
    """

    args = parse_args()
    output = create_status_map(
        args.status_json,
        output=args.output,
        natural_earth_resolution=args.natural_earth_resolution,
        active_color=args.active_color,
        inactive_color=args.inactive_color,
        permanently_off_color=args.permanently_off_color,
        water_color=args.water_color,
        add_borders=args.add_borders,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
