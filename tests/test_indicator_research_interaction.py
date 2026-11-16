from types import SimpleNamespace

import numpy as np
import pandas as pd
from ipympl.backend_nbagg import FigureCanvas
from matplotlib import rc_context
from matplotlib.figure import Figure

from indicator_research.interaction import attach_ohlcv_interactions


def make_interaction():
    figure = Figure(figsize=(12, 8), dpi=100)
    FigureCanvas(figure)
    axes = figure.subplots(3, 1, sharex=True)
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=20, freq="B"),
            "open": np.arange(20) + 10.0,
            "high": np.arange(20) + 11.0,
            "low": np.arange(20) + 9.0,
            "close": np.arange(20) + 10.5,
            "volume": np.arange(20) * 1000 + 1_000_000,
        }
    )
    for ax in axes:
        ax.plot(data.index, data["close"])
        ax.set_xlim(-2, len(data) + 1)
    interaction = attach_ohlcv_interactions(figure, data)
    return figure, axes, data, interaction


def test_cursor_tracks_nearest_session_across_all_axes_and_hides_on_leave():
    figure, axes, _data, interaction = make_interaction()

    interaction.on_motion(SimpleNamespace(inaxes=axes[1], xdata=4.2))

    assert interaction.hover_index == 4
    for line in interaction.cursor_lines:
        assert line.get_visible()
        np.testing.assert_array_equal(line.get_xdata(), [4, 4])

    interaction.on_leave()
    assert interaction.hover_index is None
    assert all(not line.get_visible() for line in interaction.cursor_lines)


def test_click_updates_and_moves_ohlcv_card_and_close_reopens():
    figure, axes, data, interaction = make_interaction()
    first = SimpleNamespace(inaxes=axes[0], xdata=4.2, x=150, y=200, button=1)
    interaction.on_click(first)

    first_position = (interaction.panel.get_x(), interaction.panel.get_y())
    assert interaction.selected_index == 4
    assert interaction.panel.get_visible()
    assert interaction.close_axes.get_visible()
    assert "2026-01-07" in interaction.info_text.get_text()
    assert f"开盘  {data['open'].iloc[4]:.2f}" in interaction.info_text.get_text()
    assert "成交量  1,004,000" in interaction.info_text.get_text()

    interaction.on_click(SimpleNamespace(inaxes=axes[2], xdata=9.4, x=500, y=100, button=1))
    assert interaction.selected_index == 9
    assert "2026-01-14" in interaction.info_text.get_text()
    assert first_position != (interaction.panel.get_x(), interaction.panel.get_y())

    interaction.close()
    assert interaction.selected_index is None
    assert not interaction.panel.get_visible()
    assert not interaction.info_text.get_visible()
    assert not interaction.close_axes.get_visible()

    interaction.on_click(first)
    assert interaction.panel.get_visible()
    assert interaction.selected_index == 4
    assert attach_ohlcv_interactions(figure, data) is interaction


def test_outside_click_is_ignored_and_edge_clicks_keep_card_inside_canvas():
    figure, axes, _data, interaction = make_interaction()
    interaction.on_click(SimpleNamespace(inaxes=axes[0], xdata=4, x=150, y=200, button=1))
    original_text = interaction.info_text.get_text()
    original_position = (interaction.panel.get_x(), interaction.panel.get_y())

    interaction.on_click(SimpleNamespace(inaxes=None, xdata=None, x=500, y=500, button=1))
    assert interaction.info_text.get_text() == original_text
    assert original_position == (interaction.panel.get_x(), interaction.panel.get_y())

    interaction.on_click(SimpleNamespace(
        inaxes=axes[-1], xdata=19, x=figure.bbox.width - 1, y=figure.bbox.height - 1, button=1
    ))
    assert interaction.panel.get_x() >= .01
    assert interaction.panel.get_y() >= .01
    assert interaction.panel.get_x() + interaction.panel_width <= .99
    assert interaction.panel.get_y() + interaction.panel_height <= .99


def test_ohlcv_text_inherits_configured_chinese_font():
    with rc_context({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"]}):
        *_unused, interaction = make_interaction()

    assert interaction.info_text.get_fontfamily() == ["Noto Sans CJK JP", "DejaVu Sans"]
