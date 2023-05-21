from __future__ import annotations

import itertools
from msilib import sequence
from typing import Any, Literal, Optional, Sequence, Type, TypeVar, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import PolyCollection
from sklearn.neighbors import KernelDensity


def get_my_rcparams(linewidth: int = 5, major_size=20) -> dict[str, dict[str, Any]]:
    rc_params = {
        "figure": {
            "figsize": (15, 15),
            "subplot.left": 0.15,
        },
        "font": {
            "family": "Arial",
            "size": 45,
            "weight": "bold",
        },
        "axes": {
            "titleweight": "bold",
            "spines.right": False,
            "spines.top": False,
            "linewidth": linewidth,
            "labelweight": "bold",
        },
        "xtick": {
            "major.width": linewidth,
            "major.size": major_size,
            "major.pad": 12,
            "direction": "in",
        },
        "ytick": {
            "major.width": linewidth,
            "major.size": major_size,
            "direction": "in",
            "major.pad": 12,
        },
        "legend": {"fancybox": False, "markerscale": 2.0},
    }
    return rc_params


T = TypeVar("T", int, float)


def waterfall_plot(
    ax_3d,
    dim_2_array: np.ndarray,
    extent: tuple[T, T, T, T],
    edgecolors: str | Sequence[str] = "k",
    fill: Optional[bool] = False,
    facecolors: str | Sequence[str] = "w",
    alpha: float | int = 1,
    zmin: float | int = 0,
):
    (
        y_len,
        x_len,
    ) = dim_2_array.shape
    x_min, x_max, y_min, y_max = extent
    xs = np.linspace(x_min, x_max, x_len)
    ys = np.linspace(y_min, y_max, y_len)
    verts: list[Any] = []
    ims: list[Any] = []
    for ydir in range(y_len):
        zs = dim_2_array[ydir]
        if fill:
            verts.append(
                list(zip(np.hstack((xs[0], xs, xs[-1])), np.hstack((zmin, zs, zmin))))
            )
        else:
            im = ax_3d.plot(
                xs,
                np.full(xs.shape, ys[ydir]),
                zs,
                c=edgecolors
                if type(edgecolors) is str or edgecolors is None
                else edgecolors[ydir],
            )
            ims.append(*im)
    if fill:
        polygon = PolyCollection(
            verts, edgecolors=edgecolors, facecolors=facecolors, alpha=alpha
        )
        ims = ax_3d.add_collection3d(polygon, zs=ys, zdir="y")
        ax_3d.set(
            xlim=[x_min, x_max],
            ylim=[y_min, y_max],
            zlim=[dim_2_array.min(), dim_2_array.max()],
        )
    return ims


KwsType = dict[Literal["bar_kw", "err_kw", "scatter_kw"], dict[str, Any]]


def error_plot(
    dataset: Sequence[Sequence[int | float]],
    ax,
    bar_xs: Sequence[int] | None = None,
    colors: str | Sequence[str] = "k",
    scatter_shift_nums: float | Sequence[float] = 0.1,
    error_type: Literal["SD", "SE"] = "SD",
    custom_kws: Optional[KwsType] = None,
) -> None:  # datasetの次元に注意．
    """
    各データの平均棒グラフ，散布図，エラーバーを同時に描画する．

    Example:
    for group, values in rc_params.items(): plt.rc(group, **values)
    sample_dataset = np.random.rand(2, 15)
    fig, axes = plt.subplots(1, 2)
    error_plot(sample_dataset, axes[0], colors='k', error_type='SE', scatter_shift_nums=0.5, custom_kws={
        'bar_kw':{
            'hatch':['//', '**']
        },
        'scatter_kw':{
            'edgecolor':'k',
            'linewidth':2
        },
        'err_kw':{
            'capsize':50
        }
    })
    error_plot(sample_dataset, axes[1], error_type='SE')
    for ax, title in zip(axes, ('custom', 'default')): ax.set(title=title, xlabel='data±SE', xticklabels=('data1', 'data2'))
    fig.tight_layout()
    plt.show()
    """
    row_num = len(dataset)
    col_nums = [len(arr) for arr in dataset]  # 行ごとの列の数
    flat = itertools.chain.from_iterable

    if not isinstance(scatter_shift_nums, float):
        assert len(dataset) == len(scatter_shift_nums)
    else:
        scatter_shift_nums = [scatter_shift_nums] * row_num  # キャスト

    if bar_xs is None:
        bar_xs = range(row_num)
    else:
        assert len(bar_xs) == row_num

    scatter_xs = tuple(
        flat(
            [
                [x + shift_num] * x_len
                for x, x_len, shift_num in zip(bar_xs, col_nums, scatter_shift_nums)
            ]
        )
    )  # 散布図はscatter_shift_numだけずらすとエラーバーが見やすい
    scatter_colors = (
        tuple(flat([[color] * x_len for color, x_len in zip(colors, col_nums)]))
        if not isinstance(colors, str)
        else colors
    )
    aves = [
        np.mean(arr) for arr in dataset
    ]  # 行ごとにの列の数が異なる可能性があるため，それぞれの行に分解してmean, stdを使う必要がある．np.mean(dataset, axis=1)だとエラー
    errors = [np.std(arr) for arr in dataset]
    if error_type == "SE":
        errors = [err / np.sqrt(x_len) for err, x_len in zip(errors, col_nums)]

    # デフォルトの各プロットオプション
    bar_kw = {
        "edgecolor": colors,
        "fill": False,
        "color": colors,
        "linewidth": 5,
    }
    err_kw = {
        "yerr": errors,
        "capsize": 0,
        "capthick": 8,
        "elinewidth": 8,
        "fmt": "none",
        "ecolor": "k",
        "lolims": True,
    }
    scatter_kw = {"color": scatter_colors, "s": 150}

    # custom_kwsが設定されたときはキーワードの追加または上書き
    if custom_kws is not None:
        bar_kw = (
            {**bar_kw, **custom_kws["bar_kw"]} if "bar_kw" in custom_kws else bar_kw
        )
        err_kw = (
            {**err_kw, **custom_kws["err_kw"]} if "err_kw" in custom_kws else err_kw
        )
        scatter_kw = (
            {**scatter_kw, **custom_kws["scatter_kw"]}
            if "scatter_kw" in custom_kws
            else scatter_kw
        )

    ax.bar(bar_xs, aves, **bar_kw)
    plotline, caplines, barlinecols = ax.errorbar(bar_xs, aves, **err_kw)
    ax.scatter(scatter_xs, tuple(flat(dataset)), **scatter_kw)
    ax.set(xticks=bar_xs)
    caplines[0].set_marker("_")
    caplines[0].set_markersize(25)


def plot_3d_spectrogram(
    ax_3d, array: np.ndarray, N: int, fs: float, window_size: int, step: int
) -> None:
    freq_mesh = [fs * k / window_size for k in range(window_size // 2 + 1)]
    time_mesh = np.linspace(0, N * (1 / fs), (N - window_size) // step)
    X, Y = np.meshgrid(time_mesh, freq_mesh)
    Z = array.T
    ax_3d.plot_surface(X, Y, Z, cmap="terrain")


def scatter_hist(
    xs: Sequence[Sequence[int | float]],
    ys: Sequence[Sequence[int | float]],
    colors: Sequence[str] | str | None = None,
    labels: Sequence[str] | None = None,
    ratio: int = 8,
    kernel: bool | str = False,
    bandwidth: float | int = 5.0,
    kernel_kw: dict[str, Any] = None,
    hist_kw: dict[str, Any] | None = None,
    scatter_kw: dict[str, Any] | None = None,
):
    """
    ヒストグラム付きの散布図．内部で figure インスタンスを作成し, figと[散布図, x方向ヒストグラム, y方向ヒストグラム]のaxesを返す.

    xs, ys : データの配列 (行数:ラベルの数, 列数: データの数)
    colors : データの行数と等しい文字配列または文字列
    labels : データの行数と等しいラベルの文字配列
    ratio : 散布図とヒストグラムの比率 (散布図のサイズ:ヒストグラムのサイズ = ratio : 1)
    kernel : True でカーネル密度推定を描画 'both'を渡すとヒストグラムと重ねて描画
    bandwidth : カーネル密度推定のバンド幅
    kernel_kw : カーネル密度推定の描画に渡したいオプション
    hist_kw : ヒストグラムの描画に渡したいオプション
    scatter_kw : 散布図の描画に渡したいオプション


    Example:
    x = np.random.rand(100); y = np.random.rand(100)
    x_2 = np.random.rand(100); y_2 = np.random.rand(100)

    xs = [x, x_2]; ys = [y, y_2]
    colors = ['k', 'r']
    hist_kw = {
        'bins':100,
        'alpha':0.4
    }
    scatter_kw = {
        's':100,
        'edgecolor':'w'
    }
    fig, ax = scatter_hist(xs, ys, colors, ['Negative', 'Positive'], True, hist_kw, scatter_kw)
    ax.set(xlabel='x', ylabel='y', )
    plt.legend(bbox_to_anchor=(1., 1.05), fontsize=24, loc='lower left')
    plt.show()
    """
    hist_default_kw = {
        "bins": 100,
        "alpha": 0.4,
    }

    scatter_default_kw = {}

    kernel_default_kw = {}

    if hist_kw is not None:
        hist_kw = {**hist_default_kw, **hist_kw}
    else:
        hist_kw = hist_default_kw
    if scatter_kw is not None:
        scatter_kw = {**scatter_default_kw, **scatter_kw}
    else:
        scatter_kw = scatter_default_kw
    if kernel_kw is not None:
        kernel_kw = {**kernel_default_kw, **kernel_kw}
    else:
        kernel_kw = kernel_default_kw

    fig = plt.figure()
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(ratio, 1),
        height_ratios=(1, ratio),
    )
    hist_x = fig.add_subplot(gs[0, 0:-1])
    hist_y = fig.add_subplot(gs[1:, -1])
    kde_x_ax = hist_x.twinx()
    kde_y_ax = hist_y.twiny()
    scatter = fig.add_subplot(gs[1:, 0:-1])
    if colors is None or isinstance(colors, str):
        colors = [colors] * len(xs)
    if labels is None:
        labels = [labels] * len(xs)
    flat_x = [el for x in xs for el in x]
    flat_y = [el for y in ys for el in y]
    xmin, xmax = min(flat_x), max(flat_x)
    ymin, ymax = min(flat_y), max(flat_y)
    for x, y, color, label in zip(xs, ys, colors, labels):
        x: np.ndarray = np.array(x)
        y: np.ndarray = np.array(y)
        scatter.scatter(x, y, label=label, c=color, **scatter_kw)
        if kernel:
            kde_x = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
                np.array(x)[:, None]
            )
            kde_y = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
                np.array(y)[:, None]
            )
            dens_x = np.linspace(xmin, xmax, 500)[:, None]
            dens_y = np.linspace(ymin, ymax, 500)[:, None]
            kde_x_ax.plot(
                dens_x,
                np.exp(kde_x.score_samples(dens_x)) * x.shape[0],
                color=color,
                **kernel_kw,
            )
            kde_y_ax.plot(
                np.exp(kde_y.score_samples(dens_y)) * x.shape[0],
                dens_y,
                color=color,
                **kernel_kw,
            )
        if not kernel or kernel == "both":
            hist_x.hist(x, color=color, **hist_kw)
            hist_y.hist(y, orientation="horizontal", color=color, **hist_kw)
    for ax in (hist_x, hist_y, kde_x_ax, kde_y_ax):
        for direction in ("right", "top", "left", "bottom"):
            ax.spines[direction].set_visible(False)
            ax.tick_params(**{direction: False, f"label{direction}": False})
    for ax in (hist_x, kde_x_ax):
        ax.set(xlim=scatter.get_xlim())
        ax.spines["bottom"].set_visible(True)
    for ax in (hist_y, kde_y_ax):
        ax.set(ylim=scatter.get_ylim())
        ax.spines["left"].set_visible(True)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    axes = [scatter, hist_x, hist_y]
    return fig, axes


def abbreviated_plot(
    xs: Sequence[Sequence],
    ys: Sequence[Sequence],
    interrupt_x: float | int,
    reopen_x: float | int,
    colors: Sequence[str] | str | None = None,
    labels: Sequence[str] | None = None,
    plot_kw: dict[str, Any] | None = None,
):
    """ ":
    x軸を途中省略した折れ線グラフを描画する.内部で figure インスタンスを作成し, figと[前半のデータ, 後半]のaxesを返す.

    xs, ys : データの配列 (行数:ラベルの数, 列数: データの数)
    interrupt_x:省略開始地点のX
    reopen_x:省略後のx
    colors : データの行数と等しい文字配列または文字列
    labels : データの行数と等しいラベルの文字配列
    plot_kw : プロット描画時に渡したいオプション引数

    Example:
    xs = np.linspace(1, 1000, 1000); ys = np.linspace(1, 1000, 1000)
    plot_kw = {'lw':3, 'linestyle':'--'}
    fig, axes = abbreviated_plot([xs], [ys], 305, 800, colors=['k'], labels=['test'], plot_kw=plot_kw)
    axes[1].set(xticks=[1000])
    plt.legend()
    plt.show()
    """
    plot_default_kw = {}

    if plot_kw is not None:
        plot_kw = {**plot_default_kw, **plot_kw}
    else:
        plot_kw = plot_default_kw

    fig, axes = plt.subplots(ncols=2, sharey="row", width_ratios=(2, 1))
    axes[1].tick_params(left=False, labelleft=False)
    over_y = 0.01
    height = 0.05
    wnum = 35  # 奇数
    pp = (0, height, 0, -height)
    px = np.array([pp[i % 4] for i in range(0, wnum)])
    py = np.linspace(-1 * over_y, 1 + over_y, wnum)
    p = Path(list(zip(px, py)), [Path.MOVETO] + [Path.CURVE3] * (wnum - 1))

    line1 = mpatches.PathPatch(
        p,
        lw=13,
        edgecolor="black",
        facecolor="None",
        clip_on=False,
        transform=axes[1].transAxes,
        zorder=10,
    )
    line2 = mpatches.PathPatch(
        p,
        lw=8,
        edgecolor="white",
        facecolor="None",
        clip_on=False,
        transform=axes[1].transAxes,
        zorder=10,
        capstyle="round",
    )
    axes[1].add_patch(line1)
    axes[1].add_patch(line2)
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(left=False, labelleft=False)
    if colors is None or isinstance(colors, str):
        colors = [colors] * len(xs)
    if labels is None:
        labels = [labels] * len(xs)
    for i, (x, y, color, label) in enumerate(zip(xs, ys, colors, labels)):
        for ax in axes:
            ax.plot(x, y, color=color, label=label, **plot_kw)
        xmin, xmax = axes[0].get_xlim()
        axes[0].set(xlim=[xmin, interrupt_x])
        if i == 0:
            axes[1].set(xlim=[reopen_x, xmax])
    fig.subplots_adjust(wspace=0.00)
    return fig, axes


if __name__ == "__main__":

    xs = np.linspace(1, 1000, 1000)
    ys = np.linspace(1, 1000, 1000)
    plot_kw = {"lw": 3, "linestyle": "--"}
    fig, axes = abbreviated_plot(
        [xs], [ys], 305, 800, colors=["k"], labels=["test"], plot_kw=plot_kw
    )
    axes[1].set(xticks=[1000])
    plt.legend()
    plt.show()
