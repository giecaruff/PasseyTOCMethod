# -*- coding: utf-8 -*-

from __future__ import print_function

import codecs
import json
import os

import dash
from dash import Dash, Input, Output, State, dcc, html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

import las2
import readcsv


_DEPTHSHIFT = 345.0
_CONFIG_PATH = "configuration.json"
_OUTPUT_CSV = "toc.csv"

_PLOTLY_COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "yellow": "#bcbd22",
    "cyan": "#17becf",
}


def getdisplayname(name, unit):
    if unit is None:
        return name.strip()
    unit = unit.strip()
    if unit:
        return "{} ({})".format(name.strip(), unit)
    return name.strip()


def mergelogs(logs):
    mergedlog = np.empty(len(logs[0]))
    mergedlog[:] = np.nan

    for log in logs:
        where = np.isfinite(log)
        mergedlog[where] = log[where]

    return mergedlog


def baselinedatatolog(depth, baseline):
    log = np.empty(depth.shape[0], dtype=float)
    log[:] = np.nan
    log[:] = baseline
    return log


def passeymethod(dt, logrt, dtbaseline, logrtbaseline, lom):
    dlogrt = (logrt - logrtbaseline) + 0.02 * (dt - dtbaseline)
    toc = dlogrt * 10 ** (2.297 - 0.1688 * lom)
    return np.clip(toc, 0.0, 100.0)


def load_configuration():
    with open(_CONFIG_PATH, "r") as f:
        return json.load(f)


def load_lab_data(config):
    labconfig = dict(config["labdata"])
    rows = dict(labconfig.pop("rows"))
    csvfilename = labconfig.pop("filename")

    with codecs.open(csvfilename, encoding="latin-1") as csvfile:
        _csvheader, csvdata = readcsv.readcsv(csvfile, **labconfig)

    wells = np.array(csvdata[rows.pop("well")], dtype=str)
    labdata = {}

    for wellname in np.unique(wells):
        labdata[wellname] = {}
        where = wells == wellname
        for key, index in rows.items():
            labdata[wellname][key] = np.array(csvdata[index], dtype=float)[where]

    return labdata


def load_log_data(config):
    logconfig = dict(config["logdata"])
    lasfilesdir = logconfig.pop("lasfilesdir")
    mnemonics = dict(logconfig.pop("mnemoics"))
    depthmnem = mnemonics.pop("depth")

    logdata = {}

    for lasfilename in sorted(os.listdir(lasfilesdir)):
        if not lasfilename.lower().endswith(".las"):
            continue

        lasfile = las2.LAS2Parser(os.path.join(lasfilesdir, lasfilename))
        wellname = os.path.splitext(lasfilename)[0]

        welldata = {
            "depth": {
                "name": depthmnem,
                "unit": lasfile.data[depthmnem]["unit"],
                "data": lasfile.data[depthmnem]["values"].copy(),
                "displayname": getdisplayname(depthmnem, lasfile.data[depthmnem]["unit"]),
            }
        }

        for curvekey, mnemonic in mnemonics.items():
            curvedata = {
                "name": mnemonic,
                "unit": None,
                "data": [],
                "displayname": mnemonic,
            }

            for curvename, curve in lasfile.data.items():
                if not curvename.startswith(mnemonic):
                    continue
                if curvedata["unit"] is None:
                    curvedata["unit"] = curve["unit"]
                    curvedata["displayname"] = curvename
                curvedata["data"].append(curve["values"].copy())

            if len(curvedata["data"]) > 1:
                curvedata["data"] = mergelogs(curvedata["data"])
            elif len(curvedata["data"]) == 1:
                curvedata["data"] = curvedata["data"][0]
            else:
                raise KeyError("Curve '{}' was not found in {}.".format(mnemonic, lasfilename))

            welldata[curvekey] = curvedata

        logdata[wellname] = welldata

    return logdata


def color_for(name):
    return _PLOTLY_COLORS.get(name, name)


def prepare_well_data(wellname, config, labdata, logdata):
    limits = config["visualization"]["limits"]
    initial = config["initialparameters"]
    smoothing = config["others"]["smoothing"]
    resampling = config["others"]["resampling"]

    depth = logdata[wellname]["depth"]["data"].astype(float).copy() - _DEPTHSHIFT
    labdepth = labdata[wellname]["top"].astype(float).copy() - _DEPTHSHIFT

    dt = logdata[wellname]["dt"]["data"].astype(float).copy()
    rt = logdata[wellname]["rt"]["data"].astype(float).copy()
    gr = logdata[wellname]["gr"]["data"].astype(float).copy()
    cali = logdata[wellname]["cali"]["data"].astype(float).copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        logrt = np.log10(rt)

    finite = np.isfinite(dt) & np.isfinite(logrt) & np.isfinite(gr) & np.isfinite(cali)
    if not np.any(finite):
        raise ValueError("Well '{}' has no finite DT, RT, GR and CALI overlap.".format(wellname))

    shallow = float(np.min(depth[finite]))
    deep = float(np.max(depth[finite]))

    if smoothing["smooth"]:
        windowtype = smoothing["window"]
        if isinstance(windowtype, list):
            windowtype = tuple(windowtype)
        windowdata = signal.windows.get_window(windowtype, smoothing["windowsize"], False)
        windowdata /= np.sum(windowdata)
        dt2 = np.convolve(dt, windowdata, "same")
        logrt2 = np.convolve(logrt, windowdata, "same")
    else:
        dt2 = dt.copy()
        logrt2 = logrt.copy()

    if resampling["resample"]:
        depth2 = np.linspace(deep, shallow, resampling["npoints"])
        dt2 = np.interp(depth2, depth, dt2)
        logrt2 = np.interp(depth2, depth, logrt2)
    else:
        depth2 = depth.copy()

    caliplot = (
        (limits["gr"][1] - limits["gr"][0])
        * (cali - limits["cali"][0])
        / (limits["cali"][1] - limits["cali"][0])
        + limits["gr"][0]
    )

    return {
        "wellname": wellname,
        "depth": depth,
        "depth2": depth2,
        "dt": dt,
        "dt2": dt2,
        "rt": rt,
        "logrt": logrt,
        "logrt2": logrt2,
        "gr": gr,
        "cali": cali,
        "caliplot": caliplot,
        "labdepth": labdepth,
        "labtoc": labdata[wellname]["toc"].astype(float).copy(),
        "shallow": shallow,
        "deep": deep,
        "dtbaseline": float(initial["dtbaseline"]),
        "logrtbaseline": float(initial["logrtbaseline"]),
        "lom": float(initial["lom"]),
    }


def line_trace(x, y, name, color, xaxis=None):
    trace = go.Scattergl(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line={"color": color, "width": 1.5},
        hovertemplate="%{x:.3f}<br>Depth %{y:.2f}<extra>" + name + "</extra>",
    )
    if xaxis:
        trace.update(xaxis=xaxis)
    return trace


def build_log_figure(well, config, baseline_state, lom, depth_range):
    limits = config["visualization"]["limits"]
    colors = config["visualization"]["colors"]
    logdata = APP_DATA["logdata"]

    shallow, deep = sorted([float(depth_range[0]), float(depth_range[1])])
    dtbaseline = float(baseline_state["dt"])
    logrtbaseline = float(baseline_state["logrt"])

    dtbl = baselinedatatolog(well["depth2"], dtbaseline)
    logrtbl = baselinedatatolog(well["depth2"], logrtbaseline)
    toc = passeymethod(well["dt2"], well["logrt2"], dtbl, logrtbl, float(lom))

    pd.DataFrame({"DEPT": np.flip(well["depth2"]), "TOC": np.flip(toc)}).to_csv(
        _OUTPUT_CSV, index=False
    )

    dt_delta = -0.02 * (well["dt"] - dtbaseline)
    logrt_delta = well["logrt"] - logrtbaseline

    fig = make_subplots(
        rows=1,
        cols=5,
        shared_yaxes=True,
        horizontal_spacing=0.015,
        column_widths=[0.18, 0.2, 0.2, 0.2, 0.22],
        subplot_titles=(
            "{} / {}".format(
                logdata[well["wellname"]]["gr"]["displayname"],
                logdata[well["wellname"]]["cali"]["displayname"],
            ),
            "{} baseline".format(logdata[well["wellname"]]["dt"]["displayname"]),
            "log({}) baseline".format(logdata[well["wellname"]]["rt"]["displayname"]),
            "Delta log R",
            "TOC",
        ),
    )

    fig.add_trace(
        line_trace(well["gr"], well["depth"], "GR", color_for(colors["gr"])), row=1, col=1
    )
    fig.add_trace(
        line_trace(well["caliplot"], well["depth"], "CALI", color_for(colors["cali"])),
        row=1,
        col=1,
    )
    fig.add_trace(
        line_trace(well["dt"], well["depth"], "DT", color_for(colors["dt"])), row=1, col=2
    )
    fig.add_trace(
        line_trace(well["logrt"], well["depth"], "log(ILD)", color_for(colors["logrt"])),
        row=1,
        col=3,
    )
    fig.add_trace(
        line_trace(dt_delta, well["depth"], "-0.02 x DT", color_for(colors["dt"])),
        row=1,
        col=4,
    )
    fig.add_trace(
        line_trace(logrt_delta, well["depth"], "log RT", color_for(colors["logrt"])),
        row=1,
        col=4,
    )
    fig.add_trace(
        line_trace(toc, well["depth2"], "Calculated TOC", color_for(colors["toc"])),
        row=1,
        col=5,
    )
    fig.add_trace(
        go.Scatter(
            x=well["labtoc"],
            y=well["labdepth"],
            mode="markers",
            name="Measured TOC",
            marker={"color": color_for(colors["labtoc"]), "size": 8},
            hovertemplate="%{x:.3f}<br>Depth %{y:.2f}<extra>Measured TOC</extra>",
        ),
        row=1,
        col=5,
    )

    baseline_color = color_for(colors["baseline"])
    fig.add_shape(
        type="line",
        x0=dtbaseline,
        x1=dtbaseline,
        y0=shallow,
        y1=deep,
        xref="x2",
        yref="y2",
        line={"color": baseline_color, "width": 4},
        editable=True,
    )
    fig.add_shape(
        type="line",
        x0=logrtbaseline,
        x1=logrtbaseline,
        y0=shallow,
        y1=deep,
        xref="x3",
        yref="y3",
        line={"color": baseline_color, "width": 4},
        editable=True,
    )

    fig.update_xaxes(range=limits["gr"], row=1, col=1, title_text="GR")
    fig.update_xaxes(range=limits["dt"], row=1, col=2, title_text="DT")
    fig.update_xaxes(range=limits["logrt"], row=1, col=3, title_text="log(ILD)")
    fig.update_xaxes(range=limits["dtlogr"], row=1, col=4, title_text="Delta log R")
    fig.update_xaxes(range=limits["toc"], row=1, col=5, title_text="TOC (%)")

    for idx in range(1, 6):
        fig.update_yaxes(range=[deep, shallow], row=1, col=idx)

    fig.update_layout(
        template="plotly_white",
        margin={"l": 12, "r": 12, "t": 54, "b": 42},
        height=760,
        dragmode="pan",
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        uirevision="passey-dashboard",
    )

    fig.update_yaxes(title_text=logdata[well["wellname"]]["depth"]["displayname"], row=1, col=1)

    return fig


def build_depth_marks(shallow, deep):
    marks = {}
    for value in np.linspace(shallow, deep, 7):
        marks[int(round(value))] = str(int(round(value)))
    return marks


def normalize_baseline_state(well, baseline_state):
    if baseline_state is None:
        return {"dt": well["dtbaseline"], "logrt": well["logrtbaseline"]}
    return {
        "dt": float(baseline_state.get("dt", well["dtbaseline"])),
        "logrt": float(baseline_state.get("logrt", well["logrtbaseline"])),
    }


def apply_shape_edits(relayout_data, baseline_state):
    if not relayout_data:
        return baseline_state

    updated = dict(baseline_state)
    shape_to_key = {0: "dt", 1: "logrt"}

    for shape_idx, key in shape_to_key.items():
        values = []
        for attr in ("x0", "x1"):
            relayout_key = "shapes[{}].{}".format(shape_idx, attr)
            if relayout_key in relayout_data:
                values.append(float(relayout_data[relayout_key]))

        nested_shapes = relayout_data.get("shapes")
        if isinstance(nested_shapes, list) and len(nested_shapes) > shape_idx:
            shape = nested_shapes[shape_idx]
            if isinstance(shape, dict):
                for attr in ("x0", "x1"):
                    if attr in shape:
                        values.append(float(shape[attr]))

        if values:
            updated[key] = float(np.mean(values))

    return updated


CONFIG = load_configuration()
APP_DATA = {
    "config": CONFIG,
    "labdata": load_lab_data(CONFIG),
    "logdata": load_log_data(CONFIG),
}
AVAILABLE_WELLS = [
    wellname for wellname in sorted(APP_DATA["logdata"].keys()) if wellname in APP_DATA["labdata"]
]

if not AVAILABLE_WELLS:
    raise RuntimeError("No LAS wells with matching lab data were found.")


app = Dash(__name__)
app.title = "Passey TOC Method"

first_well = prepare_well_data(AVAILABLE_WELLS[0], CONFIG, APP_DATA["labdata"], APP_DATA["logdata"])

app.layout = html.Div(
    [
        dcc.Store(id="baseline-store"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Well", className="control-label"),
                        dcc.Dropdown(
                            id="well-dropdown",
                            options=[{"label": well, "value": well} for well in AVAILABLE_WELLS],
                            value=AVAILABLE_WELLS[0],
                            clearable=False,
                        ),
                    ],
                    className="well-selector",
                ),
                html.Div(id="baseline-readout", className="readout"),
            ],
            className="topbar",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Depth", className="panel-title"),
                        dcc.RangeSlider(
                            id="depth-range",
                            min=first_well["shallow"],
                            max=first_well["deep"],
                            value=[first_well["shallow"], first_well["deep"]],
                            marks=build_depth_marks(first_well["shallow"], first_well["deep"]),
                            step=0.5,
                            vertical=True,
                            verticalHeight=690,
                            tooltip={"placement": "right", "always_visible": False},
                        ),
                    ],
                    className="depth-panel",
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="main-graph",
                            className="main-graph",
                            config={
                                "displaylogo": False,
                                "scrollZoom": True,
                                "editable": True,
                                "edits": {"shapePosition": True},
                                "modeBarButtonsToAdd": ["drawline", "eraseshape"],
                            },
                        ),
                        html.Div(
                            [
                                html.Div("LOM", className="panel-title"),
                                dcc.Slider(
                                    id="lom-slider",
                                    min=0,
                                    max=20,
                                    value=first_well["lom"],
                                    step=0.1,
                                    marks={i: str(i) for i in range(0, 21, 2)},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                            ],
                            className="lom-panel",
                        ),
                    ],
                    className="main-panel",
                ),
            ],
            className="dashboard",
        ),
    ],
    className="app-shell",
)

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * { box-sizing: border-box; }
            body {
                margin: 0;
                background: #f4f6f8;
                color: #1f2933;
                font-family: Arial, Helvetica, sans-serif;
            }
            .app-shell {
                min-height: 100vh;
                padding: 14px;
            }
            .topbar {
                align-items: center;
                background: #ffffff;
                border: 1px solid #d8dee6;
                border-radius: 8px;
                display: flex;
                gap: 20px;
                justify-content: space-between;
                margin-bottom: 12px;
                padding: 10px 14px;
            }
            .well-selector {
                align-items: center;
                display: grid;
                gap: 8px;
                grid-template-columns: auto 220px;
            }
            .control-label,
            .panel-title {
                color: #52616f;
                font-size: 13px;
                font-weight: 700;
                text-transform: uppercase;
            }
            .readout {
                color: #35495e;
                font-size: 14px;
                text-align: right;
            }
            .dashboard {
                display: grid;
                gap: 12px;
                grid-template-columns: 104px minmax(0, 1fr);
            }
            .depth-panel,
            .lom-panel,
            .main-panel {
                background: #ffffff;
                border: 1px solid #d8dee6;
                border-radius: 8px;
            }
            .depth-panel {
                align-items: center;
                display: flex;
                flex-direction: column;
                gap: 10px;
                padding: 12px 8px;
            }
            .main-panel {
                min-width: 0;
                overflow: hidden;
            }
            .main-graph {
                height: 760px;
            }
            .lom-panel {
                border-left: 0;
                border-right: 0;
                border-bottom: 0;
                border-radius: 0;
                padding: 10px 22px 18px;
            }
            @media (max-width: 900px) {
                .dashboard {
                    grid-template-columns: 1fr;
                }
                .depth-panel {
                    align-items: stretch;
                    min-height: 180px;
                }
                .topbar {
                    align-items: stretch;
                    flex-direction: column;
                }
                .readout {
                    text-align: left;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


@app.callback(
    Output("depth-range", "min"),
    Output("depth-range", "max"),
    Output("depth-range", "value"),
    Output("depth-range", "marks"),
    Output("lom-slider", "value"),
    Output("baseline-store", "data", allow_duplicate=True),
    Input("well-dropdown", "value"),
    prevent_initial_call=True,
)
def reset_controls_for_well(wellname):
    well = prepare_well_data(wellname, CONFIG, APP_DATA["labdata"], APP_DATA["logdata"])
    return (
        well["shallow"],
        well["deep"],
        [well["shallow"], well["deep"]],
        build_depth_marks(well["shallow"], well["deep"]),
        well["lom"],
        {"dt": well["dtbaseline"], "logrt": well["logrtbaseline"]},
    )


@app.callback(
    Output("baseline-store", "data"),
    Input("main-graph", "relayoutData"),
    State("well-dropdown", "value"),
    State("baseline-store", "data"),
)
def update_baseline_from_shape(relayout_data, wellname, baseline_state):
    well = prepare_well_data(wellname, CONFIG, APP_DATA["labdata"], APP_DATA["logdata"])
    baseline_state = normalize_baseline_state(well, baseline_state)
    return apply_shape_edits(relayout_data, baseline_state)


@app.callback(
    Output("main-graph", "figure"),
    Output("baseline-readout", "children"),
    Input("well-dropdown", "value"),
    Input("baseline-store", "data"),
    Input("lom-slider", "value"),
    Input("depth-range", "value"),
)
def update_graph(wellname, baseline_state, lom, depth_range):
    well = prepare_well_data(wellname, CONFIG, APP_DATA["labdata"], APP_DATA["logdata"])
    baseline_state = normalize_baseline_state(well, baseline_state)
    if depth_range is None:
        depth_range = [well["shallow"], well["deep"]]

    figure = build_log_figure(well, CONFIG, baseline_state, lom, depth_range)
    readout = "DT baseline: {:.3f} | log(ILD) baseline: {:.3f} | LOM: {:.1f}".format(
        baseline_state["dt"], baseline_state["logrt"], float(lom)
    )
    return figure, readout


if __name__ == "__main__":
    print("Starting Passey TOC dashboard.")
    print("Open http://127.0.0.1:8050 in your browser.")
    app.run(debug=False, host="127.0.0.1", port=8050)
