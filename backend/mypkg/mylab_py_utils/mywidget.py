from __future__ import annotations

import os
import tkinter as tk
from abc import ABCMeta, abstractmethod
from glob import glob
from tkinter import filedialog
from typing import Callable, Container, Sequence, TypeVar

import cv2
import numpy as np
from PIL import Image, ImageTk

Value = TypeVar("Value")


class ContainerManager(tk.LabelFrame, metaclass=ABCMeta):  # 抽象クラス
    def __init__(
        self,
        master,
        contents: Container[Value],
        receive_content_func: Callable[[tk.Event, Value], None],
        side: str = "top",
        **kw,
    ):
        """
        receive_content_funcに与えられる引数はイベントとcontents内の要素.
        継承先の_create_content_widget内で作成したウィジェットのバインド時に設定する.
        """
        super().__init__(master, **kw)
        self.contents: Container[Value] = contents
        self.receive_content_func: Callable[[Value], None] = receive_content_func
        self.side: str = side
        self.update()

    def update(
        self,
    ):  # クラスの外側やreceive_content_func内でcontentsを更新したときはこのメソッドを呼ぶとウィジェットが更新される
        for widget in self.winfo_children():
            widget.destroy()
        for content in self.contents:
            frame = tk.Frame(self)
            self._create_content_widget(frame, content)
            frame.pack(side=self.side)

    @abstractmethod
    def _create_content_widget(self, frame: tk.Frame, content: Value) -> None:
        ...  # リストの中の表示するコンテンツの説明や操作ボタンをこの抽象メソッドで決める．


class ImageCanvas(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.img_id: int | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.img: np.ndarray | None = None

    def update_img(self, img: np.ndarray):
        if self.photo is not None:
            self.delete(self.img_id)
        self.img = img
        self.photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), master=self
        )
        self.img_id = self.create_image(0, 0, anchor="nw", image=self.photo)
        self["height"], self["width"], _ = img.shape


class ScrollImageViewer(tk.LabelFrame, metaclass=ABCMeta):
    def __init__(self, master: tk.Widget, **kw):
        super().__init__(master, **kw)
        self.viewer = ImageCanvas(self)
        self.option = ViewOptionSelector(self, text="option")
        self.viewer.bind("<MouseWheel>", self._viewer_cmd)
        self.index: int = 0
        self._update()
        for w in (self.option, self.viewer):
            w.pack()

    def _viewer_cmd(self, e: tk.Widget) -> None:
        if e.delta > 0:
            if (self.index + self.option.skip) >= self.img_len:
                self.index = self.img_len - 1
            else:
                self.index += self.option.skip
        else:
            if (self.index - self.option.skip) <= 0:
                self.index = 0
            else:
                self.index -= self.option.skip
        self._update()

    def _update(self):
        self.viewer.update_img(self.img)
        self["text"] = str(self.num)

    @property
    def num(self) -> int:
        return self.index + 1

    @property
    @abstractmethod
    def img(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def img_len(self) -> int:
        ...


class LiveLoadingViewer(ScrollImageViewer):
    def __init__(
        self,
        master: tk.Widget,
        img_paths: Sequence[str],
        add_drawing_func: None | Callable[[np.ndarray, int], np.ndarray] = None,
        **kw,
    ):
        self.img_paths: Sequence[str] = img_paths
        self.add_drawing_func: None | Callable[
            [np.ndarray, int], np.ndarray
        ] = add_drawing_func
        super().__init__(master, **kw)

    @property
    def img(self) -> np.ndarray:
        if self.add_drawing_func is None:
            return cv2.imread(self.img_paths[self.index])
        else:
            return self.add_drawing_func(
                cv2.imread(self.img_paths[self.index]), self.index
            )

    @property
    def img_len(self) -> int:
        return len(self.img_paths)


class CompletedLoadingViewer(ScrollImageViewer):
    def __init__(self, master: tk.Widget, imgs: Sequence[np.ndarray], **kw):
        self.imgs: Sequence[np.ndarray] = imgs
        super().__init__(master, **kw)

    @property
    def img(self) -> np.ndarray:
        return self.imgs[self.index]

    @property
    def img_len(self) -> int:
        return len(self.imgs)


class ViewOptionSelector(tk.LabelFrame):
    def __init__(self, master: ScrollImageViewer, **kw):
        super().__init__(master, **kw)
        self.master: ScrollImageViewer
        self.skip: int = 1
        self._create_widget()

    def _create_widget(self) -> None:
        def jump_cmd(e: tk.Event):
            self.master.index = int(jump_entry.get()) - 1
            self.master._update()

        def skip_cmd(e: tk.Event):
            self.skip = int(skip_entry.get())

        jump_area = tk.Frame(self)
        tk.Label(jump_area, text=f"jump (1~{self.master.img_len})").pack()
        jump_entry = tk.Entry(jump_area, justify="center")
        jump_entry.bind("<Return>", jump_cmd)
        skip_area = tk.Frame(self)
        tk.Label(skip_area, text="skip").pack()
        skip_entry = tk.Entry(skip_area, justify="center")
        skip_entry.bind("<Return>", skip_cmd)
        for w in (jump_entry, skip_entry):
            (w.insert(0, str(1)), w.pack())
        for w in (jump_area, skip_area):
            w.pack(side=tk.LEFT)


class ImageExtractor(tk.Tk):
    def __init__(self, image: np.ndarray, **kw):
        super().__init__(**kw)
        self.origin_img: np.ndarray = image
        height, width, _ = image.shape
        self.viewer: ImageCanvas = ImageCanvas(self, width=width, height=height)
        self.viewer.update_img(image)
        self.viewer.bind("<1>", lambda e: self._send_two_cor(e, self._extract_img))
        self.viewer.pack()
        self.subviewer: ImageCanvas = ImageCanvas(self)
        self.subviewer.pack()
        self.result: np.ndarray | None = None

    def _send_two_cor(
        self,
        e: tk.Event,
        send_func: Callable[[tuple[int, int]], None],
        *,
        __clicked_cor: list[None | int] = [None],
    ):
        if __clicked_cor[0] is None:
            __clicked_cor[0] = (e.x, e.y)
        else:
            second_clicked_cor = (e.x, e.y)
            if __clicked_cor[0] == second_clicked_cor:
                return
            send_func(__clicked_cor[0], second_clicked_cor)
            __clicked_cor[0] = None

    def _extract_img(self, cor_1: tuple[int, int], cor_2: tuple[int, int]):
        xmin, ymin = cor_1
        xmax, ymax = cor_2
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        img_copy = self.origin_img.copy()
        visualized_img = cv2.rectangle(img_copy, cor_1, cor_2, (0, 0, 0), 3)
        self.viewer.update_img(visualized_img)
        extracted_img = self.origin_img[ymin:ymax, xmin:xmax]
        self.subviewer.update_img(extracted_img)
        self.subviewer["height"], self.subviewer["width"], _ = extracted_img.shape
        if self.result is None:
            tk.Button(self, text="完了", command=self.destroy).pack()
        self.result = extracted_img


class ScrollFrame(tk.Frame):
    def __init__(self, master, x=True, y=True, custom_width=0, custom_height=0, **kw):
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        super().__init__(master, **kw)
        self._f = tk.Frame(self)
        if custom_width:
            canvas = tk.Canvas(self._f, width=custom_width, height=custom_height)
        else:
            canvas = tk.Canvas(self._f)
        self._scrollable_frame = tk.Frame(canvas)
        self._scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all"),
            ),
        )
        canvas.create_window(
            0,
            0,
            anchor="nw",
            window=self._scrollable_frame,
        )
        if y:
            self.scrollbar_y = tk.Scrollbar(
                self, orient="vertical", command=canvas.yview
            )
            self.scrollbar_y.pack(side=tk.RIGHT, fill="y")
            canvas.configure(yscrollcommand=self.scrollbar_y.set)
        if x:
            self.scrollbar_x = tk.Scrollbar(
                self, orient="horizontal", command=canvas.xview
            )
            self.scrollbar_x.pack(side=tk.BOTTOM, fill="x")
            canvas.configure(xscrollcommand=self.scrollbar_x.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.bind("<MouseWheel>", _on_mousewheel, "+")

    @property
    def f(self):  # 子ウィジェットを作るときは、子ウィジェットのインスタンス生成時、引数masterにこのプロパティを渡すこと．
        return self._scrollable_frame

    def pack(self, **kw):
        self._f.pack()
        super().pack(**kw)


def color_change_hover(event):
    if event.type == "7":
        event.widget["bg"] = "gray"
    if event.type == "8":
        event.widget["bg"] = "SystemButtonFace"


if __name__ == "__main__":

    base_dir = filedialog.askdirectory()
    img_paths = glob(os.path.join(base_dir, "*.jpg"))
    assert img_paths
    win = tk.Tk()
    viewer = LiveLoadingViewer(win, img_paths).pack()

    """"
    ↑　or 
    viewer = CompletedLoadingViewer(win, [cv2.imread(path) for path in img_paths]).pack()
    """

    win.mainloop()
