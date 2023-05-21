from __future__ import annotations

import os
import pickle
from typing import Generator, Iterator, Literal, Sequence, TypeVar

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation, FuncAnimation
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import interpolate

from .myplot import waterfall_plot


def STFT(
    array: np.ndarray, window_size: int, step: int, window_func=np.hamming
) -> np.ndarray:
    result = []
    step = step
    for i in range((len(array) - window_size) // step):
        tmp = array[i * step : i * step + window_size]
        tmp = tmp * window_func(window_size)
        amp = ((abs(DFT(tmp))) ** 2)[: window_size // 2 + 1]
        result.append(amp)
    return np.array(result)


def DFT(array):
    N = len(array)
    A = np.arange(N)
    M = np.exp(-1j * A.reshape(1, -1) * A.reshape(-1, 1) * 2 * np.pi / N)
    return np.sum(array * M, axis=1)


def IDFT(array):
    N = len(array)
    A = np.arange(N)
    M = np.exp(1j * A.reshape(1, -1) * A.reshape(-1, 1) * 2 * np.pi / N)
    return np.sum(array * M, axis=1) / N


def my_get_peakindex(array, num: int) -> np.ndarray:
    result = []
    cnt = 0
    while True:
        idxes = np.argsort(array)[::-1]
        if array[idxes[cnt]] - array[idxes[cnt] - 1] > 0:
            result.append([idxes[cnt]])
            if len(result) == num:
                break
        cnt = cnt + 1
    return np.array(result)


def get_Rectangular_face(cors: list):
    cor_1, cor_2, cor_3, cor_4, cor_5, cor_6, cor_7, cor_8 = cors
    vert_1 = np.array([cor_1] + [cor_2] + [cor_3] + [cor_4])
    vert_2 = np.array([cor_5] + [cor_6] + [cor_7] + [cor_8])
    vert_3 = np.array([cor_1] + [cor_2] + [cor_6] + [cor_5])
    vert_4 = np.array([cor_3] + [cor_4] + [cor_8] + [cor_7])
    vert_5 = np.array([cor_1] + [cor_4] + [cor_8] + [cor_5])
    vert_6 = np.array([cor_2] + [cor_3] + [cor_7] + [cor_6])
    verts = np.vstack(([vert_1], [vert_2], [vert_3], [vert_4], [vert_5], [vert_6]))
    return verts


def cross_correlate(
    f: np.ndarray,
    g: np.ndarray,
    times: np.ndarray,
    mode="full",
    animation_axes=None,
    normalize=False,
):
    if mode not in ["full", "right", "valid"]:
        raise ValueError("error")
    if animation_axes is not None:
        animation_axes[0].plot(times, f, "black")
    if normalize:
        f_mean = np.mean(f[~np.isnan(f)])
        g_mean = np.mean(g[~np.isnan(g)])
    f_N = len(f)
    g_N = len(g)
    lag_times = (
        np.array(list(reversed(-1 * times[1:g_N])) + list(times))
        if mode == "full"
        else times
    )
    range_ = (
        range(f_N + g_N - 2)
        if mode == "full"
        else range(g_N - 1, f_N + g_N - 2)
        if mode == "right"
        else range(g_N - 1, f_N - 1)
    )
    CC = []
    img_list = []
    is_move = False
    move = 0
    for tau in range_:
        if tau < g_N - 1:
            tmp_f = f[: tau + 1].copy()
            tmp_g = g[(g_N - 1) - tau :].copy()
        elif tau <= f_N - 1:
            is_move = True
            tmp_f = f[move : tau + 1].copy()
            tmp_g = g.copy()
        elif tau > f_N - 1:
            tmp_f = f[move:].copy()
            tmp_g = g[: len(tmp_f)].copy()

        if np.any(np.isnan(tmp_f)) or np.any(np.isnan(tmp_g)):
            f_nan_idx = np.where(np.isnan(tmp_f))
            g_nan_idx = np.where(np.isnan(tmp_g))
            tmp_f[np.hstack((f_nan_idx, g_nan_idx))] = np.nan
            tmp_g[np.hstack((f_nan_idx, g_nan_idx))] = np.nan
            result = (
                np.sum(tmp_f[~np.isnan(tmp_f)] * tmp_g[~np.isnan(tmp_g)])
                if not normalize
                else (
                    np.sum(
                        (tmp_f[~np.isnan(tmp_f)] - f_mean)
                        * (tmp_g[~np.isnan(tmp_g)] - g_mean)
                    )
                )
                / (
                    np.linalg.norm(tmp_f[~np.isnan(tmp_f)] - f_mean)
                    * np.linalg.norm(tmp_g[~np.isnan(tmp_g)] - g_mean)
                )
            )
        else:
            result = (
                np.sum(tmp_f * tmp_g)
                if not normalize
                else (np.sum((tmp_f - f_mean) * (tmp_g - g_mean)))
                / (np.linalg.norm(tmp_f - f_mean) * np.linalg.norm(tmp_g - g_mean))
            )
        CC.append(result)
        if animation_axes is not None:
            tmp_t = times[: tau + 1]
            y_1 = list([*[np.nan] * move]) + list(tmp_f)
            y_2 = list([*[np.nan] * move]) + list(tmp_g)
            im = animation_axes[0].plot(tmp_t, y_1, "red")
            im_2 = animation_axes[0].plot(tmp_t, y_2, "blue")
            lag_title = animation_axes[0].text(
                0.5,
                1.01,
                f"tau = {tau-(g_N-1)}",
                ha="center",
                va="bottom",
                transform=animation_axes[0].transAxes,
                fontsize="large",
            )
            im_3 = animation_axes[1].plot(lag_times[: len(CC)], CC, "black")
            img_list.append(im + im_2 + im_3 + [lag_title])
        if is_move:
            move = move + 1
    return (np.array(CC), img_list) if animation_axes is not None else np.array(CC)


def save_STAC_animation(
    times: np.ndarray,
    array: np.ndarray,
    window: int,
    step: int,
    output_dir: str,
    normalize=False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    window_len = len(array) // window
    split_array = np.array(
        [array[i * step : i * step + window] for i in range(window_len)]
    )
    for i, arr in enumerate(split_array):
        fig, axes = plt.subplots(3, 1, sharex=True)
        axes[0].set_title(f"Data window{i+1}")
        for j, arr_ in enumerate(split_array):
            axes[0].plot(
                times[: (j + 1) * window],
                [*[np.nan] * j * window] + list(arr_),
                "blue" if np.all(arr == arr_) else "black",
            )
        amp, ims = cross_correlate(array, arr, times, "right", axes[1:], normalize)
        axes[0].set_xlim(axes[1].get_xlim())
        axes[0].set_xlabel("Time [sec]")
        axes[1].set_xlabel("Time [sec]")
        axes[2].set_title("Correlation result")
        axes[2].set_xlabel("Lag [sec]")
        plt.tight_layout()
        anim = ArtistAnimation(fig, ims, interval=10)
        # break
        anim.save(os.path.join(output_dir, f"window{i+1}_convolition.gif"))
        plt.clf()
    # plt.show()


def STAC(
    times: Sequence[float | int],
    array: Sequence[float | int],
    window_size: int,
    step: int,
    normalize=False,
) -> np.ndarray:
    if not isinstance(times, np.ndarray):
        times = np.array(times)
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    result = []
    step = step
    for i in range((len(array) - window_size) // step):
        tmp_array = array[i * step : i * step + window_size]
        amp = cross_correlate(array, tmp_array, times, "right", normalize=normalize)
        result.append(amp)
    return np.array(result)


def my_interpolation(X: np.ndarray, Y: np.ndarray):
    Y_nan_idxes = np.where(np.isnan(Y))
    tmp_X = X.copy()
    tmp_Y = Y.copy()
    tmp_X = np.delete(tmp_X, Y_nan_idxes)
    tmp_Y = np.delete(tmp_Y, Y_nan_idxes)
    return interpolate.interp1d(tmp_X, tmp_Y, kind="cubic")


def moving_correlate(
    array1: np.ndarray,
    array2: np.ndarray,
    times: np.ndarray,
    window: int,
    animation_axes: np.ndarray = None,
    center=False,
):
    if len(array1) != len(array2):
        raise (ValueError)
    f = array1.copy()
    g = array2.copy()
    N = len(f)
    MC = []
    ims = []
    if animation_axes is not None:
        animation_axes[0].plot(times, array1, "black")
        animation_axes[0].plot(times, array2, "black")
    for tau in range(N - 1):
        range_ = slice(tau, tau + window)
        tmp_f = f[range_]
        tmp_g = g[range_]
        if np.any(np.isnan(tmp_f)) or np.any(np.isnan(tmp_g)):
            f_nan_idx = np.where(np.isnan(tmp_f))
            g_nan_idx = np.where(np.isnan(tmp_g))
            tmp_f[np.hstack((f_nan_idx, g_nan_idx))] = np.nan
            tmp_g[np.hstack((f_nan_idx, g_nan_idx))] = np.nan
            result = (
                np.corrcoef(tmp_f[~np.isnan(tmp_f)], tmp_g[~np.isnan(tmp_g)])[0][1]
                if len(times - 1) >= tau + window
                else np.nan
            )
        else:
            result = (
                np.corrcoef(tmp_f, tmp_g)[0][1]
                if len(times) - 1 >= tau + window
                else np.nan
            )
        MC.append(result)
        if animation_axes is not None:
            im = animation_axes[0].plot(
                times[: tau + window], [*[np.nan] * tau] + list(tmp_f), "red"
            )
            im_2 = animation_axes[0].plot(
                times[: tau + window], [*[np.nan] * tau] + list(tmp_g), "blue"
            )
            im_3 = animation_axes[1].plot(times[: tau + 1], MC, "k")
            im_4 = animation_axes[0].axvspan(
                times[tau],
                times[tau + window if len(times) - 1 >= tau + window else -1],
                color="lightgray",
            )
            ims.append(im + im_2 + im_3 + [im_4])
    return (MC, ims) if animation_axes is not None else MC


def checkpoint(path: str = "checkpoint.pickle"):
    """_summary_
    関数の返り値をバイナリデータ形式でキャッシュする．キャッシュ保存(読み込み)先を引数に渡す．デコレータとして使える．
    Args:
        path (str, optional): Cache storage path. Defaults to "checkpoint.pickle".

    """

    def _checkpoint(func):
        def __checkpoint(*args, **kwargs):
            if os.path.exists(path):
                with open(path, "rb") as f:
                    ret = pickle.load(f)
            else:
                ret = func(*args, **kwargs)
                with open(path, "wb") as f:
                    pickle.dump(ret, f)
            return ret

        return __checkpoint

    return _checkpoint


def moving_correlate_3d(
    array_1, array_2, times, window, axes=None, axes_3d=None, colors=None
):
    if not isinstance(array_1, np.ndarray):
        array_1 = np.array(array_1)
    if not isinstance(array_2, np.ndarray):
        array_2 = np.array(array_2)
    if axes_3d is not None:
        waterfall_plot(axes_3d[0], array_1, extent=[0, times[-1], 0, len(array_1)])
        waterfall_plot(axes_3d[0], array_2, extent=[0, times[-1], 0, len(array_1)])
        zmin, zmax = axes_3d[0].get_zlim()
        ymin, ymax = axes_3d[0].get_ylim()
    if axes is not None:
        axes[0].stackplot(times, *array_1, colors=colors)
        axes[1].stackplot(times, *array_2, colors=colors)
    dt = times[1] - times[0]
    D, N = array_1.shape
    F = array_1.copy()
    F_mean = F.mean(axis=1, keepdims=True)
    G = array_2.copy()
    G_mean = G.mean(axis=1, keepdims=True)
    MC_array = []
    ims = []
    for tau in range(N - 1):
        tmp_F = F[:, tau : tau + window]
        tmp_G = G[:, tau : tau + window]
        result = np.sum((tmp_F - F_mean) * (tmp_G - G_mean), axis=1).reshape(
            -1, 1
        ) / np.sum(
            (
                np.linalg.norm(tmp_F - F_mean, axis=1).reshape(-1, 1)
                * np.linalg.norm(tmp_G - G_mean, axis=1).reshape(-1, 1)
            ),
            axis=1,
        ).reshape(
            -1, 1
        )
        if tau == 0:
            MC_array = result.copy()
        else:
            MC_array = np.hstack((MC_array, result))
        if axes_3d is not None:
            cor_1 = [
                (dt * tau, ymin, zmin),
                (dt * (tau + window), ymin, zmin),
                (dt * (tau + window), ymax, zmin),
                (dt * tau, ymax, zmin),
            ]
            cor_2 = [(cor[0], cor[1], zmax) for cor in cor_1]
            window_box = Poly3DCollection(
                get_Rectangular_face(cor_1 + cor_2), color="gray", alpha=0.4
            )
            im = axes_3d[0].add_collection3d(window_box)
            im_2 = waterfall_plot(
                axes_3d[0],
                tmp_F,
                extent=[
                    times[tau],
                    times[tau + window if len(times) - 1 >= tau + window else -1],
                    0,
                    D,
                ],
                edgecolors="r",
            )
            im_3 = waterfall_plot(
                axes_3d[0],
                tmp_G,
                extent=[
                    times[tau],
                    times[tau + window if len(times) - 1 >= tau + window else -1],
                    0,
                    D,
                ],
                edgecolors="b",
            )
            im_4 = waterfall_plot(axes_3d[1], MC_array, extent=[0, times[tau], 0, D])
            ims.append([im] + im_2 + im_3 + im_4)
        if axes is not None:
            im = axes[0].axvspan(
                times[tau],
                times[tau + window if len(times) - 1 >= tau + window else -1],
                color="lightgray",
                alpha=0.7,
            )
            im_2 = axes[1].axvspan(
                times[tau],
                times[tau + window if len(times) - 1 >= tau + window else -1],
                color="lightgray",
                alpha=0.7,
            )
            ax_3_im_list = [
                axes[2].plot(times[: tau + 1], MC, color)[0]
                for MC, color in zip(MC_array, colors)
            ]
            ims.append([im] + [im_2] + ax_3_im_list)
    return (MC_array, ims) if axes_3d is not None or axes is not None else MC_array


def animateMatchTemplate(
    img: np.ndarray,
    extracted_img: np.ndarray,
    row_step: int = 1,
    col_step: int = 1,
    cmap="jet",
    method: int = cv2.TM_CCOEFF_NORMED,
) -> FuncAnimation:

    FrameType = tuple[int, int, np.ndarray, np.ndarray]

    def update(frame: FrameType) -> None:
        for ax in axes[1:]:
            ax.cla()
        i, j, match, img = frame
        fig.suptitle(f"i={i}, j={j}")
        axes[1].imshow(img)
        axes[2].imshow(
            match,
            extent=[0, result_w, 0, result_h],
            vmin=min_value,
            vmax=max_value,
            cmap=cmap,
        )

    def send_frames() -> Generator[tuple[int, int, np.ndarray], None, None]:
        for i in range((height - fg_height) // col_step):
            col_progress = i * col_step
            for j in range((width - fg_width) // row_step):  # 横
                row_progress = j * row_step
                bg_img = img.copy()
                bg_img[
                    col_progress : col_progress + fg_height,
                    row_progress : row_progress + fg_width,
                ] = extracted_img
                result_anim[: col_progress + 1, : row_progress + 1] = result[
                    : col_progress + 1, : row_progress + 1
                ]
                yield col_progress, row_progress, result_anim, bg_img

    fig, axes = plt.subplots(3, 1)
    cmap = get_cmap(cmap)
    height, width, _ = img.shape
    fg_height, fg_width, _ = extracted_img.shape
    result = cv2.matchTemplate(img, extracted_img, method=method)  # 結果は先に計算
    result_h, result_w = result.shape
    min_value, max_value, _, max_idx = cv2.minMaxLoc(result)
    result_anim = np.zeros((result.shape))
    left_top_x, left_top_y = max_idx
    axes[0].imshow(
        cv2.rectangle(
            img.copy(),
            (left_top_x, left_top_y),
            (left_top_x + fg_width, left_top_y + fg_height),
            (0, 0, 255),
            3,
        )
    )
    axes[0].set(xticks=[], yticks=[])
    axes[2].set(xlim=[0, result_w], ylim=[0, result_h])
    anim = FuncAnimation(fig, update, send_frames, interval=1)
    fig.colorbar(
        ScalarMappable(cmap=cmap, norm=Normalize(vmin=min_value, vmax=max_value)),
        orientation="horizontal",
    )
    return anim


U = TypeVar("U", int, float)


def scale(
    xmin: U,
    xmax: U,
    ymin: U,
    ymax: U,
    rate: float,
    basepoint: Literal[
        "center", "upper left", "upper right", "downer left", "downer right"
    ]
    | tuple[U, U] = "center",
):

    if isinstance(basepoint, str):
        if basepoint == "center":
            base_x = (xmin + xmax) / 2
            base_y = (ymin + ymax) / 2
        if basepoint == "upper left":
            base_x = xmin
            base_y = ymax
        if basepoint == "upper right":
            base_x = xmax
            base_y = ymax
        if basepoint == "downer left":
            base_x = xmin
            base_y = ymin
        if basepoint == "downer right":
            base_x = xmax
            base_y = ymin
    else:
        base_x = basepoint[0]
        base_y = basepoint[1]

    array = np.array([[xmin, xmax], [ymin, ymax], [1, 1]])
    affine_1 = np.array([[1, 0, base_x], [0, 1, base_y], [0, 0, 1]])
    affine_2 = np.array([[rate, 0, 0], [0, rate, 0], [0, 0, 1]])
    affine_3 = np.array(
        [
            [
                1,
                0,
                -base_x,
            ],
            [0, 1, -base_y],
            [0, 0, 1],
        ]
    )

    scaled_array = affine_1 @ affine_2 @ affine_3 @ array
    scaled_xmin, scaled_xmax = scaled_array[0]
    scaled_ymin, scaled_ymax = scaled_array[1]
    return scaled_xmin, scaled_xmax, scaled_ymin, scaled_ymax


def central_idxes_gen(
    container_size: int, win_num: int
) -> Generator[Iterator[int], None, None]:
    """
    中央を基準とした窓の計算に用いる．イテレート毎に移動するインデックス群のイテレータを返すジェネレータ．
    最初は前方のインデックス群，　最後は後方のインデックス群を返す．

    container_size:データのサイズ
    win_num:s左右の窓の幅. 例えばwin_num = 1を指定すると, 左, 中央, 右の3つのインデックスを返す.

    Example:
    x = np.linspace(0, 1, 100); y = np.sin(2*np.pi*x) + np.random.rand(100)
    gen = central_idxes_gen(100, 3)
    moving_aves = []
    for _ in x:
        moving_aves.append(np.average([y[i] for i in next(gen)]))
    plt.plot(x, y)
    plt.plot(x, moving_aves)
    plt.show()
    """

    index = 0
    while True:
        if index < win_num:
            sample_nums = range(index, index + (win_num * 2 + 1))
        elif index + win_num >= container_size:
            sample_nums = range(index - win_num * 2, index + 1)
        else:
            sample_nums = list(range(index - win_num, index)) + list(
                range(index, index + (win_num + 1))
            )
        yield iter(sample_nums)
        index += 1
        if index == container_size:
            index = 0
            return


if __name__ == "__main__":
    N = 256
    window = 64
    times = np.linspace(0, 10, N)
    dt = times[1] - times[0]
    freq = np.linspace(1, 2, 3).reshape(-1, 1)
    y_1_list = np.sin(2 * np.pi * times * freq)
    y_2_list = 0.5 * np.cos(2 * np.pi * times * freq)
    fig, axes = plt.subplots(2, 1, subplot_kw={"projection": "3d"})
    axes[0].set_box_aspect((3, 2, 1))
    mo, ims = moving_correlate_3d(y_1_list, y_2_list, times, window, axes)
    anim = ArtistAnimation(fig, ims)
    plt.show()
