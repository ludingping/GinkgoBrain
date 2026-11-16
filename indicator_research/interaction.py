"""Shared cursor tracking and OHLCV information cards for notebook figures."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button


class OHLCVInteraction:
    def __init__(self, figure, data: pd.DataFrame):
        self.figure = figure
        self.data = data
        self.plot_axes = list(figure.axes)
        self.hover_index = None
        self.selected_index = None
        self.cursor_lines = [
            ax.axvline(np.nan, color="#444444", linewidth=.7, linestyle="--", alpha=.7, visible=False, zorder=20)
            for ax in self.plot_axes
        ]

        figure_width, figure_height = figure.get_size_inches()
        self.panel_width = min(2.9 / figure_width, .9)
        self.panel_height = min(1.65 / figure_height, .9)
        self.pad_x = .12 / figure_width
        self.pad_y = .12 / figure_height
        self.button_width = .28 / figure_width
        self.button_height = .28 / figure_height

        self.panel = FancyBboxPatch(
            (0, 0),
            self.panel_width,
            self.panel_height,
            boxstyle="round,pad=0.004",
            transform=figure.transFigure,
            facecolor="white",
            edgecolor="#555555",
            linewidth=.8,
            alpha=.96,
            visible=False,
            zorder=1000,
        )
        figure.add_artist(self.panel)
        self.info_text = figure.text(
            0, 0, "", fontsize=9, va="top", ha="left", visible=False, zorder=1001
        )
        self.close_axes = figure.add_axes([0, 0, self.button_width, self.button_height], frame_on=False)
        self.close_axes.set_zorder(1002)
        self.close_axes.set_visible(False)
        self.close_button = Button(self.close_axes, "×", color="#f2f2f2", hovercolor="#dddddd")
        self.close_button.label.set_fontsize(12)
        self.close_cid = self.close_button.on_clicked(self.close)

        self.motion_cid = figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.click_cid = figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.leave_cid = figure.canvas.mpl_connect("figure_leave_event", self.on_leave)
        figure._ohlcv_interaction = self

    def nearest_index(self, event) -> int | None:
        if event.inaxes not in self.plot_axes or event.xdata is None:
            return None
        index = int(np.rint(event.xdata))
        return index if 0 <= index < len(self.data) else None

    def on_motion(self, event) -> None:
        index = self.nearest_index(event)
        if index == self.hover_index:
            return
        self.hover_index = index
        for line in self.cursor_lines:
            line.set_visible(index is not None)
            if index is not None:
                line.set_xdata([index, index])
        self.figure.canvas.draw_idle()

    def on_leave(self, _event=None) -> None:
        if self.hover_index is None:
            return
        self.hover_index = None
        for line in self.cursor_lines:
            line.set_visible(False)
        self.figure.canvas.draw_idle()

    def on_click(self, event) -> None:
        if event.button != 1:
            return
        index = self.nearest_index(event)
        if index is not None:
            self.show_day(index, event.x, event.y)

    def show_day(self, index: int, canvas_x: float, canvas_y: float) -> None:
        if not 0 <= index < len(self.data):
            raise IndexError("trading-day index out of range")
        self.selected_index = index
        row = self.data.iloc[index]
        date = pd.to_datetime(row["date"]) if "date" in row else pd.to_datetime(self.data.index[index])
        self.info_text.set_text(
            f"OHLCV  {date:%Y-%m-%d}\n"
            f"开盘  {row['open']:.2f}\n"
            f"最高  {row['high']:.2f}\n"
            f"最低  {row['low']:.2f}\n"
            f"收盘  {row['close']:.2f}\n"
            f"成交量  {row['volume']:,.0f}"
        )

        click_x = canvas_x / self.figure.bbox.width
        click_y = canvas_y / self.figure.bbox.height
        left = click_x + .015
        if left + self.panel_width > .99:
            left = click_x - self.panel_width - .015
        left = min(max(left, .01), .99 - self.panel_width)
        bottom = min(max(click_y - self.panel_height / 2, .01), .99 - self.panel_height)

        self.panel.set_bounds((left, bottom, self.panel_width, self.panel_height))
        self.info_text.set_position((left + self.pad_x, bottom + self.panel_height - self.pad_y))
        self.close_axes.set_position([
            left + self.panel_width - self.button_width - self.pad_x,
            bottom + self.panel_height - self.button_height - self.pad_y,
            self.button_width,
            self.button_height,
        ])
        self.panel.set_visible(True)
        self.info_text.set_visible(True)
        self.close_axes.set_visible(True)
        self.figure.canvas.draw_idle()

    def close(self, _event=None) -> None:
        self.selected_index = None
        self.panel.set_visible(False)
        self.info_text.set_visible(False)
        self.close_axes.set_visible(False)
        self.figure.canvas.draw_idle()


def attach_ohlcv_interactions(figure, data: pd.DataFrame) -> OHLCVInteraction:
    existing = getattr(figure, "_ohlcv_interaction", None)
    if existing is not None:
        return existing
    return OHLCVInteraction(figure, data)
