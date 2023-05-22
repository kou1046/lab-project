from __future__ import annotations

import json
import os
import tkinter as tk
from dataclasses import asdict, dataclass
from tkinter import messagebox
from typing import Any, Callable, Literal, Sequence
import cv2
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from tqdm import tqdm

from .intermediate_model import BoundingBox, Point
from mypkg.mylab_py_utils.mywidget import ScrollFrame, ImageCanvas, ContainerManager

"""
補完idのファイル(stop_ids, replace_ids, monitor_ids, create_ids計4つ)をComplementIdCreatorクラスによって作成するスクリプト.
補完ファイル作成後はpreprocessor.py内のcomplementerのコンストラクタに4つのファイルパスを入力してインスタンス化し, boxandcombinerの引数にcomplementerを入力する.
"""


@dataclass
class FrameWithImgBoxes:
    img_path: str
    boxes: list[BoundingBox]

    @property
    def img(self) -> np.ndarray:
        return cv2.imread(self.img_path)

    @property
    def ids(self) -> set[int]:
        return {box.id for box in self.boxes}

    def find_boxes_bypoint_inside(
        self, point: tuple[int, int], add_boxes: None | list[BoundingBox] = None
    ) -> list[BoundingBox]:
        target_boxes = self.boxes if add_boxes is None else (self.boxes + add_boxes)
        isin_boxes = [box.contains(Point(*point)) for box in target_boxes]
        return (np.array(target_boxes)[isin_boxes]).tolist()

    def find_box_byid(self, id: int, add_boxes: None | list[BoundingBox] = None) -> BoundingBox:
        target_boxes = self.boxes if add_boxes is None else (self.boxes + add_boxes)
        return [box for box in target_boxes if box.id == id][0]


@dataclass
class StopResume:  # 停止フレームと再開フレームが複数ある可能性があるため．stop_idだけ少し複雑.
    id: int
    stop_frames: list[int]
    resume_frames: list[float | int]  # float('inf')の可能性があるので型はfloat|int. float('inf')のときは再開フレームが存在せず，ずっと停止したままのidを指す
    index: int = 0
    status: Literal["resuming", "stopping"] = "resuming"

    def needsstop(self, frame) -> bool:  # このメソッドとindexプロパティはこのスクリプトでは使用しない
        if self.index > len(self.stop_frames) - 1:  # 最後のresumeをすでに使用している
            return False
        stop_frame = self.stop_frames[self.index]
        resume_frame = self.resume_frames[self.index]
        if frame >= stop_frame and frame < resume_frame:  # 停止フレーム期間内
            self.status = "stopping"
            return True
        else:
            if self.status == "stopping":  # 直前まで止まっていた時
                self.status = "resuming"
                self.index += 1
            return False


@dataclass(frozen=True)
class TrackingBoundingBox(BoundingBox):  # createidsの記録に使う，開始するフレーム情報と，動的に動いた軌跡を格納するtrackingプロパティを追加している．
    start_frame: int
    istracking: bool = False

    def __post_init__(self):
        self._replace_range(Point(self.min.x, self.min.y), Point(self.max.x, self.max.y))

    def _replace_range(self, min: Point, max: Point) -> None:  # lazy
        object.__setattr__(self, "tracking_min", min)
        object.__setattr__(self, "tracking_max", max)

    def update_tracking_range_from_img(self, prev_img: np.ndarray, new_img: np.ndarray) -> None:
        """
        prev_img内ボックス画像の'一部'を抜き取り, new_imgのボックス内画像と相互相関によるテンプレートマッチングを行い, 類似度を計算する.
        類似度が一番高かったテンプレートの座標を取得し, 計算前と計算後の座標移動変化をtrackingプロパティに反映する.
        """

        prev_self_img = prev_img[
            self.tracking_min.y : self.tracking_max.y,
            self.tracking_min.x : self.tracking_max.x,
        ]  # prev_imgのボックス内画像
        height, width, _ = prev_self_img.shape
        pt1, pt2 = get_scaled_rectangle(
            (0, 0), (width, height), 0.5, (width // 2, height // 2)
        )  # prev_self_imgにおける，ボックス真ん中から1/2の距離をトリミングした座標
        prev_left_x, prev_top_y = pt1
        prev_right_x, prev_btm_y = pt2
        prev_template_img = prev_self_img[prev_top_y:prev_btm_y, prev_left_x:prev_right_x]  # テンプレート

        new_self_img = new_img[
            self.tracking_min.y : self.tracking_max.y,
            self.tracking_min.x : self.tracking_max.x,
        ]  # new_imgのボックス内画像
        cc_matrix = cv2.matchTemplate(new_self_img, prev_template_img, cv2.TM_CCOEFF_NORMED)

        min_v, max_v, min_idx, max_idx = cv2.minMaxLoc(cc_matrix)
        new_left_x, new_top_y = max_idx
        delta_x = new_left_x - prev_left_x
        delta_y = new_top_y - prev_top_y

        height, width, _ = prev_img.shape

        new_min = Point((self.tracking_min.x + delta_x), (self.tracking_min.y + delta_y))
        new_max = Point((self.tracking_max.x + delta_x), (self.tracking_max.y + delta_y))
        self._replace_range(new_min, new_max)


class MonitorIdsManager(ContainerManager):
    def __init__(self, master, contents: set[int], receive_func: Callable[[int], None], **kw):
        self.grand: ComplementIdCreator = master.nametowidget(".")
        self.contents: set[int]
        self.canvases: list[ImageCanvas] = []
        super().__init__(master, contents, receive_func, **kw)

    def _create_content_widget(self, frame: tk.Frame, content: int):
        id_ = content
        f = tk.LabelFrame(frame, text=f"ID:{id_}")
        canvas = ImageCanvas(f, width=100, height=100)
        if id_ in self.grand.completed_ids:
            replacing, replaced_id = self.grand._isreplacing(id_)
            if replacing:
                box = self.grand._completed_find_box_byid(replaced_id)
            else:
                box = self.grand._completed_find_box_byid(id_)
            person_img = self.grand.viewed_frame.img[box.min.y : box.max.y, box.min.x : box.max.x]
        else:
            person_img = np.zeros((100, 100, 3), dtype=np.uint8)
        canvas.update_img(cv2.resize(person_img, dsize=(100, 100)))
        canvas.person_id = id_  # この追加プロパティはクリックしたときにidを送信するときにも利用 → _send_clicked_id
        canvas.pack()
        del_btn = tk.Button(f, text="-")
        del_btn.bind("<1>", lambda e: [self.receive_content_func(e, id_), self.update()])
        del_btn.pack()
        f.pack(side=tk.LEFT)
        self.canvases.append(canvas)

    def update(self):
        self.canvases = []
        super().update()

    def photo_update(
        self,
    ) -> set[int]:  # 毎回すべて生成しなおすのは重いので 画像のみ更新したいときはこのメソッド(マウススクロール時に用いる) 返り値があり，更新されたIDを返す
        return_values: set[int] = set()
        for canvas in self.canvases:
            isreplacing, replaced_id = self.grand._isreplacing(
                canvas.person_id
            )  # canvas.person_idはmonitor_idのどれか replaced_idはmonitor_idに置きかわっている最新のid
            if (
                isreplacing and canvas.person_id in self.grand.viewed_frame.ids
            ):  # どれかのidをmonitor_idに置き換えた後に真のmonitor_idが出現するとバグるので認識させないでおく（要調整）
                canvas.master["bg"] = "black"
            elif canvas.person_id in self.grand.completed_ids:
                canvas.master["bg"] = "#d9d9d9"
                if isreplacing:
                    self._canvas_update(canvas, replaced_id)
                    canvas.master["text"] = f"ID:{canvas.person_id} ({replaced_id})"
                    return_values.add(replaced_id)
                else:
                    self._canvas_update(canvas, canvas.person_id)
                    return_values.add(canvas.person_id)
            elif self.grand._needsstop(canvas.person_id):
                canvas.master["bg"] = "green"  # 停止中なら緑
                if isreplacing and replaced_id in self.grand.viewed_frame.ids:
                    self._canvas_update(canvas, replaced_id)
                elif canvas.person_id in self.grand.viewed_frame.ids:
                    self._canvas_update(canvas, canvas.person_id)
            else:
                canvas.master["bg"] = "red"  # どれにも該当しない（現在見ているフレームに存在しない）なら赤
        return return_values

    def _canvas_update(self, canvas: ImageCanvas, id_: int) -> None:
        box = self.grand._completed_find_box_byid(id_)
        person_img = self.grand.viewed_frame.img[box.min.y : box.max.y, box.min.x : box.max.x]
        canvas.update_img(cv2.resize(person_img, dsize=(100, 100)))


class StopIdsManager(ContainerManager):
    def __init__(self, master, contents: StopIds, receive_func: Callable[[int], None], **kw):
        self.grand: ComplementIdCreator = master.nametowidget(".")
        super().__init__(master, contents, receive_func, **kw)

    def _create_content_widget(self, frame: tk.Frame, content: int) -> None:
        def cancel():  # 直前の停止を除去
            del value.resume_frames[-1]
            del value.stop_frames[-1]
            value.status = "resuming"
            if not self.contents[id_].stop_frames:
                del self.contents[id_]

        def resume():  # 停止から再開
            if value.stop_frames[-1] >= self.grand.frame_num:  # 直前の停止フレームより今見ているフレームのほうが小さいとき
                return
            value.status = "resuming"
            value.resume_frames[-1] = self.grand.frame_num  # infから現在のフレームに
            self.grand._viewer_update(None)

        id_ = content
        value: StopResume = self.contents[id_]
        id_frame = tk.LabelFrame(frame, text=f"ID:{id_}")
        del_btn = tk.Button(id_frame, text="-")
        del_btn.bind("<1>", lambda e: [self.receive_content_func(e, id_), self.update()])
        del_btn.pack(side=tk.LEFT)
        for stop_frame, resume_frame in zip(value.stop_frames, value.resume_frames):
            tk.Label(id_frame, text=f"{stop_frame}以降停止 → {resume_frame}以降再開").pack()
        if value.status == "stopping":  # もし停止中なら"再開"ボタンと直前の停止をキャンセルするボタンを追加する
            f = tk.Frame(id_frame)
            tk.Button(f, text="キャンセル", command=lambda: [cancel(), self.update(), print(value)]).pack(side=tk.LEFT)
            resume_btn = tk.Button(
                f,
                text="再開",
            )
            resume_btn.bind(
                "<1>",
                lambda e: [resume(), resume_btn.destroy(), self.update(), print(value)],
            )
            resume_btn.pack(side=tk.LEFT)
            f.pack()
        id_frame.pack(side=tk.LEFT)


class ReplaceIdsManager(ContainerManager):
    def __init__(self, master, contents: ReplaceIds, receive_func: Callable[[int], None], **kw):
        self.grand: ComplementIdCreator = master.nametowidget(".")
        super().__init__(master, contents, receive_func, **kw)

    def _create_content_widget(self, frame: tk.Frame, content: int) -> None:
        frame_num = content
        value: dict[int, int] = self.contents[frame_num]
        for done_id, do_id in value.items():
            tk.Label(frame, text=f"フレーム{frame_num}以降{done_id} → {do_id}").pack(side=tk.LEFT)
        del_btn = tk.Button(frame, text="-")
        del_btn.bind("<1>", lambda e: [self.receive_content_func(e, frame_num), self.update()])
        del_btn.pack()


class CreateIdsManager(ContainerManager):
    def __init__(
        self,
        master,
        contents: CreateIds,
        receive_func: Callable[[TrackingBoundingBox], None],
        **kw,
    ):
        self.grand: ComplementIdCreator = master.nametowidget(".")
        super().__init__(master, contents, receive_func, **kw)

    def _create_content_widget(self, frame: tk.Frame, content: TrackingBoundingBox) -> None:
        istracking_str = "有効" if content.istracking else "無効"
        tk.Label(
            frame,
            text=(f"{content.start_frame}フレーム 以降{content.id}追加 追跡：" + istracking_str),
        ).pack(side=tk.LEFT)
        del_btn = tk.Button(frame, text="-")
        del_btn.bind(
            "<1>",
            lambda e: [
                self.receive_content_func(e, content),
                self.update(),
                self.grand._viewer_update(None),
            ],
        )
        del_btn.pack()


class OptionSelector(tk.LabelFrame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.grand: ComplementIdCreator = self.nametowidget(".")
        self.skip: int = 1
        self.isvisible: bool = False
        self._create_widget()

    def _create_widget(self):
        def update(e: tk.Event):
            if e.widget is skip_frame_entry:
                self.skip = int(skip_frame_entry.get())
                skip_explanation.set(f"フレームスキップ(現在{self.skip})")
            if e.widget is jump_frame_entry:
                self.grand.viewer.index_ = int(jump_frame_entry.get()) - 1
                self.grand._viewer_update(None)
            if e.widget is toggle:
                self.isvisible = isvisible_var.get()
                self.grand._viewer_update(None)

        def next_jump_error_frame():  # 次のid飛びがあるフレームまでジャンプ
            fake_scroll = tk.Event()
            fake_scroll.delta = 1  # フェイクのマウススクロールイベントインスタンスを作成し，自動的にスクロールさせる
            completed_monitor_ids = self.grand.completed_ids
            self.grand.option_selector.skip = 1
            while True:
                self.grand._viewer_update(fake_scroll)
                if completed_monitor_ids != self.grand.completed_ids or self.grand.frame_num == len(self.grand.frames):
                    break
                completed_monitor_ids = self.grand.completed_ids

        isvisible_var = tk.BooleanVar(self)
        isvisible_var.set(True)
        toggle = tk.Checkbutton(self, variable=isvisible_var, text="ボックスを描画しない")
        jump_frame_area = tk.Frame(self)
        jump_frame_entry = tk.Entry(jump_frame_area, justify=tk.CENTER)
        tk.Label(jump_frame_area, text=f"指定したフレームに移動 1~{len(self.grand.frames)}").pack()
        skip_frame_area = tk.Frame(self)
        skip_explanation = tk.StringVar(skip_frame_area)
        skip_explanation.set(f"フレームスキップ(現在{self.skip})")
        skip_frame_entry = tk.Entry(skip_frame_area, justify=tk.CENTER)
        tk.Label(skip_frame_area, textvariable=skip_explanation).pack()
        skip_frame_entry.bind("<Return>", update)
        jump_frame_entry.bind("<Return>", update)
        toggle.bind("<1>", update)
        toggle.pack(side=tk.LEFT)
        jump_frame_entry.pack()
        skip_frame_entry.pack()
        jump_frame_area.pack(side=tk.LEFT)
        skip_frame_area.pack(side=tk.LEFT)
        tk.Button(
            self,
            text=" jump next error frame \n(Experimental)",
            command=next_jump_error_frame,
        ).pack(side=tk.LEFT, padx=20)


class ClikedPersonViewer(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas: ImageCanvas | None = None

    def create_image(self, person_img: np.ndarray):
        main_f = tk.LabelFrame(self, text="clicked")
        self.canvas = ImageCanvas(main_f, width=100, height=100)
        self.canvas.update_img(cv2.resize(person_img, dsize=(100, 100)))
        self.canvas.pack()
        main_f.pack(side=tk.LEFT)
        tk.Label(self, text="Who ?\n→", font=("", 25)).pack(side=tk.LEFT, padx=10)

    def forget(self):
        self.canvas = None
        for widget in self.winfo_children():
            widget.destroy()


monitor_filename = "monitor_ids.json"
replace_filename = "replace_ids.json"
stopresume_filename = "stopresume_ids.json"
create_filename = "create_ids.json"
StopIds = dict[int, StopResume]
ReplaceIds = dict[int, dict[int, int]]
MonitorIds = set[int]
CreateIds = list[TrackingBoundingBox]


class ComplementIdCreator(tk.Tk):
    total_frame = 1

    def __init__(self, id_csv_paths: Sequence[str], img_paths: Sequence[str], out_dir: str):
        super().__init__()
        if os.path.exists(os.path.join(out_dir, "complements")):
            (
                self.monitor_ids,
                self.replace_ids,
                self.stop_ids,
                self.create_ids,
            ) = deserialize_complement_ids(
                os.path.join(out_dir, "complements", monitor_filename),
                os.path.join(out_dir, "complements", replace_filename),
                os.path.join(out_dir, "complements", stopresume_filename),
                os.path.join(out_dir, "complements", create_filename),
            )
            if self.create_ids:
                self.create_num: int = min([box.id for box in self.create_ids]) - 1
            else:
                self.create_num = 0

        else:
            self.stop_ids: StopIds = {}
            self.replace_ids: ReplaceIds = {}
            self.monitor_ids: MonitorIds = set()
            self.create_ids: CreateIds = []
            self.create_num: int = -1
        self.frames: list[FrameWithImgBoxes] = []
        self.out_dir: str = out_dir
        height, width, _ = cv2.imread(img_paths[0]).shape
        for csv_path, img_path in tqdm(zip(id_csv_paths, img_paths), total=len(id_csv_paths)):
            self.frames.append(
                FrameWithImgBoxes(
                    img_path,
                    [
                        BoundingBox(
                            id_,
                            Point(xmin, ymin),
                            Point(
                                xmax if xmax <= width else width,
                                ymax if ymax <= height else height,
                            ),
                        )
                        for id_, xmin, ymin, xmax, ymax in read_id_csv(csv_path)
                    ],
                )
            )
        scroll_frame = ScrollFrame(self, custom_height=1600, custom_width=1600)
        self.main_f = tk.LabelFrame(scroll_frame.f, text=ComplementIdCreator.total_frame)
        replace_stop_area = tk.Frame(self.main_f)
        self.option_selector: OptionSelector = OptionSelector(self.main_f, text="Option")
        init_img = self.frames[0].img
        height, width, _ = init_img.shape
        self.viewer: ImageCanvas = ImageCanvas(self.main_f, width=width, height=height)
        self.viewer.update_img(init_img)
        self.viewer.index_ = 0  # ImageCanvasにindex_プロパティを追加している
        self.sub_viewer_frame = tk.Frame(self.main_f)
        self.clicked_person_viewer = ClikedPersonViewer(self.sub_viewer_frame)
        self.monitor_list: MonitorIdsManager = MonitorIdsManager(
            self.sub_viewer_frame,
            self.monitor_ids,
            lambda e, c: self.monitor_ids.remove(c),
            text="monitor",
            side="left",
        )
        self.replace_list: ReplaceIdsManager = ReplaceIdsManager(
            replace_stop_area,
            self.replace_ids,
            lambda e, c: self.replace_ids.pop(c),
            text="replace",
            side="bottom",
        )
        self.stop_list: StopIdsManager = StopIdsManager(
            replace_stop_area,
            self.stop_ids,
            lambda e, c: self.stop_ids.pop(c),
            text="stop",
            side="bottom",
        )
        self.create_list: CreateIdsManager = CreateIdsManager(
            replace_stop_area,
            self.create_ids,
            lambda e, c: self.create_ids.remove(c),
            text="create",
            side="bottom",
        )
        self.main_f.pack()
        scroll_frame.pack()
        self.option_selector.pack()
        self.viewer.pack()
        self.clicked_person_viewer.pack(side=tk.LEFT, padx=10)
        self.monitor_list.pack(side=tk.LEFT)
        self.sub_viewer_frame.pack()
        replace_stop_area.pack()
        self.stop_list.pack(side=tk.LEFT)
        self.replace_list.pack(side=tk.LEFT)
        self.create_list.pack(side=tk.LEFT)
        self.protocol("WM_DELETE_WINDOW", self._exit)
        self._create_btns()
        self.viewer.bind("<Button-4>", self._viewer_update, "+")
        self.viewer.bind("<Button-5>", self._viewer_update, "+")

    def _create_btns(self):
        def btn_cmd(e: tk.Event, pressed_name: str):  # ボタンを押されたときの処理．クリックバインドを有効化する．
            for btn in btn_f.winfo_children():
                btn.pack_forget()
            btn_f["text"] = pressed_name
            explanation = tk.StringVar(btn_f)
            tk.Label(btn_f, textvariable=explanation).pack()
            if pressed_name == "replace":
                explanation.set("同一人物の異なるID(エラーID)を選択してください．")
                click_binds = [
                    self.viewer.bind(
                        "<1>",
                        lambda e: self._send_two_clicked_ids(e, self._add_replace_id),
                    ),
                    *[
                        widget.bind(
                            "<1>",
                            lambda e: self._send_two_clicked_ids(e, self._add_replace_id),
                        )
                        for widget in self.monitor_list.canvases
                    ],
                ]  # replaceだけ2回クリック待ち
            if pressed_name == "stop":
                explanation.set("監視を一時停止するIDを選択してください．")
                click_binds = [
                    self.viewer.bind("<1>", lambda e: self._send_clicked_id(e, self._add_stop_id)),
                    *[
                        widget.bind("<1>", lambda e: self._send_clicked_id(e, self._add_stop_id))
                        for widget in self.monitor_list.canvases
                    ],
                ]
            if pressed_name == "monitor":
                explanation.set("監視対象に加えるIDを選択してください．")
                click_binds = [self.viewer.bind("<1>", lambda e: self._send_clicked_id(e, self._add_monitor_id))]

            if pressed_name == "create":
                explanation.set("2点をクリックしてボックスを作成してください．")
                istracking_var = tk.BooleanVar(btn_f)
                tk.Checkbutton(btn_f, variable=istracking_var, text="Enable tracking").pack()
                istracking_var.set(True)
                click_binds = [
                    self.viewer.bind(
                        "<1>",
                        lambda e: self._send_two_clicked_cor(e, self._add_create_id, istracking=istracking_var.get()),
                    )
                ]

            tk.Button(
                btn_f,
                text="OK",
                command=lambda: [
                    btn_f.destroy(),
                    [self.unbind("<1>", click_bind) for click_bind in click_binds],
                    self._send_two_clicked_ids(None, self._add_replace_id),
                    self._send_two_clicked_cor(None, self._add_create_id),
                    self._create_btns(),
                    self.clicked_person_viewer.forget(),
                ],
            ).pack()

        btn_f = tk.LabelFrame(self, text="operation")
        for text, bindkey in [
            ("replace", "<Control-r>"),
            ("stop", "<Control-s>"),
            ("monitor", "<Control-m>"),
            ("create", "<Control-c>"),
        ]:
            btn = tk.Button(btn_f, text=text)
            btn.bind("<1>", lambda e, arg=text: btn_cmd(e, arg))
            btn.pack(side=tk.LEFT)
            self.bind(
                bindkey,
                lambda e, arg=text: [
                    btn_cmd(e, arg),
                    self._send_two_clicked_ids(None, self._add_replace_id),
                    self._send_two_clicked_cor(None, self._add_create_id),
                    self.clicked_person_viewer.forget(),
                ],
            )
        btn_f.pack(after=self.viewer)

    def _viewer_update(self, e: tk.Event | None):  # マウススクロール時や監視id追加時などの画像の更新, Noneはスクロール時以外で更新したいときに渡す
        prev_viewer_img = self.viewer.img

        if isinstance(e, tk.Event):
            if e.num == 4 and self.viewer.index_ < len(self.frames) - 1:
                self.viewer.index_ = (
                    (self.viewer.index_ + self.option_selector.skip)
                    if (self.frame_num + self.option_selector.skip < len(self.frames))
                    else len(self.frames) - 1
                )
            if e.num == 5 and self.viewer.index_ > 0:
                self.viewer.index_ = (
                    (self.viewer.index_ - self.option_selector.skip)
                    if (self.frame_num - self.option_selector.skip >= 1)
                    else 0
                )

        updated_ids = self.monitor_list.photo_update()  # モニターidリストの画像更新と，画像更新に利用されたidの取得
        new_viewer_img = self.viewed_frame.img

        for stamped_box in self.added_box:
            if (stamped_box.id not in self.scoped_replace_dict) and (stamped_box.start_frame != self.frame_num):
                continue

            if stamped_box.istracking:  # createしたボックスが，トラッキング有効だったとき
                if stamped_box.start_frame == self.frame_num:
                    stamped_box._replace_range(stamped_box.min, stamped_box.max)
                stamped_box.update_tracking_range_from_img(
                    prev_viewer_img, new_viewer_img
                )  # createしたボックスの座標を更新する（追従する）

            pt1 = [getattr(stamped_box.min, name) for name in ("x", "y")]
            pt2 = [getattr(stamped_box.max, name) for name in ("x", "y")]
            cv2.rectangle(new_viewer_img, pt1, pt2, (255, 0, 0), 5)
            cv2.putText(
                new_viewer_img,
                f"Created ID:{self.scoped_replace_dict[stamped_box.id] if stamped_box.id in self.scoped_replace_dict else stamped_box.id}",
                pt1,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                3,
            )

        if self.option_selector.isvisible:
            boxes = [
                self._completed_find_box_byid(id_) for id_ in updated_ids if self._isreplacing(id_)[1] > 0
            ]  # 後半のifはcreateしたボックスを省いている(描画が重なってしまうため).
            for box in boxes:
                cv2.rectangle(
                    new_viewer_img,
                    (box.min.x, box.min.y),
                    (box.max.x, box.max.y),
                    (0, 0, 255),
                    5,
                )
                cv2.putText(
                    new_viewer_img,
                    f"ID:{self.scoped_replace_dict[box.id] if box.id in self.scoped_replace_dict else box.id}",
                    (box.min.x, box.min.y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )

        self.viewer.update_img(new_viewer_img)
        self.main_f["text"] = f"フレーム:{self.frame_num}"

    def _add_monitor_id(self, id_):
        if id_ in self.monitor_ids:
            self.monitor_ids.remove(id_)
        else:
            print(f"add monitor:{id_}")
            self.monitor_ids.add(id_)
        self.monitor_list.update()
        self._viewer_update(None)

    def _add_stop_id(self, id_: int):
        scoped_replace_dict = self.scoped_replace_dict
        if (id_ not in self.monitor_ids) and (id_ not in scoped_replace_dict):
            messagebox.showwarning("警告", "クリックするIDは監視の対象である必要があります.")
            return

        if id_ in scoped_replace_dict:  # ストップするのはmonitor_idのどれかなので置き換え後のid
            id_ = scoped_replace_dict[id_]

        if id_ not in self.stop_ids:  # 初めて停止させるなら
            self.stop_ids[id_] = StopResume(id_, [self.frame_num], [float("inf")], status="stopping")
        else:  # 2回目以降の停止させるなら
            if self.stop_ids[id_].status == "stopping":  # 既に停止中なら
                return
            print(f"add stop:{id_}")
            self.stop_ids[id_].status = "stopping"  # 停止させる
            self.stop_ids[id_].stop_frames.append(self.frame_num)
            self.stop_ids[id_].resume_frames.append(float("inf"))
        self.stop_list.update()
        self._viewer_update(None)

    def _add_replace_id(self, frame: int, done_id: int, do_id: int):
        if (do_id not in self.monitor_ids) and (do_id not in self.scoped_replace_dict):
            messagebox.showwarning("警告", "2回目にクリックするIDは監視の対象である必要があります.")
            return
        print(f"add replace id. frame:{frame}, done:{done_id}, do:{do_id}")
        scoped_replace_dict = self.scoped_replace_dict
        while True:
            if do_id in scoped_replace_dict:  # do_idがさらに置き換え対象である場合は，monitor_idのどれかにたどるまでreplaceを繰り返す
                do_id = scoped_replace_dict[do_id]
            else:
                break
        added_dict = {done_id: do_id}
        if self.frame_num in self.replace_ids:
            self.replace_ids[frame] = {**self.replace_ids[self.frame_num], **added_dict}
        else:
            self.replace_ids[frame] = added_dict
        self.replace_list.update()
        self._viewer_update(None)

    def _add_create_id(self, cor_1: tuple[int, int], cor_2: tuple[int, int], istracking: bool = False):
        xmin, ymin = cor_1
        xmax, ymax = cor_2
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        added_img = self.viewer.img
        self.viewer.update_img(added_img)
        added_box = TrackingBoundingBox(
            self.create_num,
            Point(xmin, ymin),
            Point(xmax, ymax),
            self.frame_num,
            istracking,
        )
        self.create_ids.append(added_box)
        self.create_list.update()
        self._viewer_update(None)
        self.create_num -= 1

    def _send_clicked_id(
        self, e: tk.Event, send_id_func: Callable[[int], Any]
    ):  # send_id_func:クリックしたときに実行する, コールバック関数．引数にはクリックしたidが渡される
        if e.widget is self.viewer:  # クリックしたウィジェットがメインビューワーだった時
            clicked_boxes = self._completed_find_boxes_bypoint_inside((e.x, e.y))
            if not clicked_boxes:
                return
            elif len(clicked_boxes) == 1:
                clicked_id = clicked_boxes[0].id
            else:
                sub_win = tk.Toplevel(self)
                sub_win.geometry(f"+{e.x}+{e.y}")
                sub_win.title = "複数のIDが見つかりました．"
                var = tk.IntVar(sub_win)
                for box in clicked_boxes:
                    tk.Radiobutton(sub_win, text=f"ID:{box.id}", value=box.id, variable=var).pack()
                var.set(box.id)
                tk.Button(sub_win, text="完了", command=sub_win.destroy).pack()
                sub_win.grab_set()
                sub_win.focus_set()
                sub_win.transient(self)
                self.wait_window(sub_win)
                clicked_id = var.get()
        else:  # モニターリストだった時（上の小さい画像）
            clicked_id = e.widget.person_id
        send_id_func(clicked_id)

    def _send_two_clicked_ids(
        self,
        e: tk.Event | None,
        send_id_func: Callable[[int, int, int], Any],  # send_id_funcに渡される引数は[最初のクリック時のフレーム，最初のクリックされたid, 2回目のクリックされたid]
        *,
        __first_clicked_id: list[int | None] = [None],
        __first_clicked_frame_num: list[int | None] = [None],
    ):  # デフォルト引数にミュータブルな型を渡したときの挙動を利用して，最初のクリックされたidとフレームを記憶している
        if e is None:  # Noneを受信すると記憶していた最初クリック時のidとフレームは初期化する
            __first_clicked_id[0] = None
            __first_clicked_frame_num[0] = None
            return

        def inner(clicked_id: int):
            if __first_clicked_id[0] is None:  # 一回目クリック時の処理
                __first_clicked_id[0] = clicked_id
                __first_clicked_frame_num[0] = self.frame_num
                clicked_box = self._completed_find_box_byid(clicked_id)

                min_point = clicked_box.min
                max_point = clicked_box.max
                clicked_person_img = self.viewed_frame.img[min_point.y : max_point.y, min_point.x : max_point.x]
                self.clicked_person_viewer.create_image(clicked_person_img)

            else:  # 二回目クリック時の処理
                second_clicked_id = clicked_id
                if second_clicked_id == __first_clicked_id[0]:
                    return
                send_id_func(
                    __first_clicked_frame_num[0],
                    __first_clicked_id[0],
                    second_clicked_id,
                )
                self.clicked_person_viewer.forget()
                __first_clicked_id[0] = None
                __first_clicked_frame_num[0] = None

        self._send_clicked_id(e, inner)

    def _send_two_clicked_cor(
        self,
        e: tk.Event | None,
        send_cor_func: Callable[[tuple[int, int], tuple[int, int]], None],
        *,
        __first_clicked_cor: list[tuple[int, int] | None] = [None],
        **arg,
    ):
        if e is None:
            __first_clicked_cor[0] = None
            return
        if __first_clicked_cor[0] is None:
            __first_clicked_cor[0] = (e.x, e.y)
        else:
            second_clicked_cor = (e.x, e.y)
            if __first_clicked_cor[0] == second_clicked_cor:
                return
            send_cor_func(__first_clicked_cor[0], second_clicked_cor, **arg)
            __first_clicked_cor[0] = None

    def _needsstop(self, id_: int) -> bool:  # 現在見ているフレームにおいて，渡されたidが停止期間中ならTrue, そうでないならFalse
        if id_ in self.stop_ids:
            stopresume = self.stop_ids[id_]
            for stop_frame, resume_frame in zip(stopresume.stop_frames, stopresume.resume_frames):
                if self.frame_num >= stop_frame and self.frame_num < resume_frame:
                    return True
        return False

    def _isreplacing(
        self, id_: int
    ) -> tuple[bool, int]:  # 現在見ているフレームにおいて，渡されたidが置き換える対象であれば，Trueと置き換えられるid, そうでなければFalseと引数のidをそのまま返す.
        scoped_replace_dict = self.scoped_replace_dict
        invert_replace_dict = {do: done for done, do in scoped_replace_dict.items()}
        if id_ in invert_replace_dict:
            return True, invert_replace_dict[id_]
        return False, id_

    def _exit(self):
        ComplementIdCreator.total_frame += len(self.frames)
        self._serialize_complement_ids()
        exit()

    def _serialize_complement_ids(self) -> None:
        directory_path = os.path.join(self.out_dir, "complements")
        os.makedirs(directory_path, exist_ok=True)
        with open(os.path.join(directory_path, monitor_filename), "w") as f:
            json.dump(list(self.monitor_ids), f)
        with open(os.path.join(directory_path, replace_filename), "w") as f:
            json.dump(self.replace_ids, f)
        for key in self.stop_ids:
            self.stop_ids[key] = asdict(self.stop_ids[key])
        with open(os.path.join(directory_path, stopresume_filename), "w") as f:
            json.dump(self.stop_ids, f)
        for i, _ in enumerate(self.create_ids):
            self.create_ids[i] = asdict(self.create_ids[i])
        with open(os.path.join(directory_path, create_filename), "w") as f:
            json.dump(self.create_ids, f)

    def _completed_find_box_byid(
        self, id_: int
    ):  # このメソッドは検索範囲にcreateしたボックスも含める　含めたくないときはself.viewed_frame.findbox_byid(id_)を使う
        return self.viewed_frame.find_box_byid(id_, self.added_box)

    def _completed_find_boxes_bypoint_inside(self, point: tuple[int, int]):  # ↑同上
        return self.viewed_frame.find_boxes_bypoint_inside(point, self.added_box)

    @property
    def scoped_replace_dict(
        self,
    ) -> dict[int, int]:  # 現在見ているフレーム以前の，replaceしたidを取得する (現在見ているフレームで置き換えてもよいid情報を取得する)
        frames = np.array(list(self.replace_ids))  # replaceを適用した時の全フレームのコンテナ np.array([int(フレーム数)])
        isscoped: Sequence[bool] = frames <= self.frame_num
        scoped_frames = np.sort(frames[isscoped])  # frames配列において，現在見ているフレーム以前のものに絞り込む
        scoped_replace_dict: dict[int, int] = {}  # {done:do}
        for scoped_frame in scoped_frames:
            invert_replace_dict = {do: done for done, do in self.replace_ids[scoped_frame].items()}
            scoped_replace_dict = {
                **scoped_replace_dict,
                **invert_replace_dict,
            }  # いったん逆転して追加

        return {
            done: do for do, done in scoped_replace_dict.items()
        }  # 最後に逆転して返すことで 辞書のvalue属性も一意にする(同じidがvalueに存在すると既知のエラーが出る)

    @property
    def completed_ids(self) -> set[int]:  # 現在見ているフレームの，監視中かつ停止中でない, 置き換え後のidを取得する
        created_ids = {stamped_box.id for stamped_box in self.added_box}
        viewed_ids = self.viewed_frame.ids | created_ids

        scoped_replace_dict = self.scoped_replace_dict
        viewed_replaced_ids = {
            scoped_replace_dict[id] if id in scoped_replace_dict else id for id in viewed_ids
        }  # 見ているフレームのidが置き換え対象ならmonitor_idのどれかに置き換える

        viewed_stopped_ids = viewed_replaced_ids & set(self.stop_ids)
        for stopped_id in viewed_stopped_ids.copy():
            if not self._needsstop(stopped_id):
                viewed_stopped_ids.remove(stopped_id)  # 今のフレームが停止期間内ではなかったらstopped_idsから省く

        viewed_replaced_ids_without_stopped = viewed_replaced_ids - viewed_stopped_ids  # replaceした後に停止中のidsを省く必要がある

        return viewed_replaced_ids_without_stopped & self.monitor_ids

    @property
    def added_box(
        self,
    ) -> list[TrackingBoundingBox]:  # 現在見ているフレームにおいて，適用してもよいcreateしたボックスを返す
        return_boxes: list[TrackingBoundingBox] = []
        for stamped_box in self.create_ids:
            if stamped_box.start_frame <= self.frame_num:
                return_boxes.append(stamped_box)
        return return_boxes

    @property
    def frame_num(self) -> int:  # 現在見ているフレーム番号を取得
        return ComplementIdCreator.total_frame + self.viewer.index_

    @property  # 現在見ているフレームの画像とバウンディングボックスを取得 self.viewed_frame.imgとself.viewer.imgの違いは，前者はボックス描画前のオリジナル画像を返し，後者は描画後の画像．
    def viewed_frame(self) -> FrameWithImgBoxes:
        return self.frames[self.viewer.index_]


def read_id_csv(csv_path: str) -> tuple[int, int, int, int, int]:
    try:
        return_values = tuple(pd.read_csv(csv_path).values.tolist())
    except EmptyDataError:
        return_values = ((0, 0, 0, 0, 0),)
    return return_values


def deserialize_complement_ids(
    monitor_json: str, replace_json: str, stop_json: str, create_json: str
) -> tuple[MonitorIds, ReplaceIds, StopIds, CreateIds]:  # jsonはキーに数字が使えないので変換するための処理が必要で少し冗長
    with open(monitor_json, "r") as f:
        monitor_ids: MonitorIds = set(json.load(f))
    with open(replace_json, "r") as f:
        tmp: dict[str, dict[str, int]] = json.load(f)
    replace_ids: replace_ids = {}
    done_do: dict[int, int] = {}
    for str_frame in tmp:
        for str_done, str_do in tmp[str_frame].items():
            done_do[int(str_done)] = int(str_do)
        replace_ids[int(str_frame)] = done_do
        done_do = {}
    with open(stop_json, "r") as f:
        tmp: dict[str, dict[str, Any]] = json.load(f)
    stopresume_ids: StopIds = {}
    for str_id in tmp:
        id_, stop_frames, resume_frames, *_ = tmp[str_id].values()
        stopresume_ids[int(str_id)] = StopResume(
            id_,
            stop_frames,
            resume_frames,
            status="stopping" if resume_frames[-1] == float("inf") else "resuming",
        )
    with open(create_json, "r") as f:
        tmp: list[dict] = json.load(f)
    create_ids: create_ids = []
    for dict_ in tmp:
        create_ids.append(TrackingBoundingBox(**dict_))
    return monitor_ids, replace_ids, stopresume_ids, create_ids


def get_scaled_rectangle(
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    level: float,
    origin_pt: tuple[int, int] = (0, 0),
) -> tuple[tuple[int, int], tuple[int, int]]:
    scaled_xmin, scaled_ymin = tuple(
        int(level * (cor - origin_cor) + origin_cor) for cor, origin_cor in zip(pt1, origin_pt)
    )
    scaled_xmax, scaled_ymax = tuple(
        int(level * (cor - origin_cor) + origin_cor) for cor, origin_cor in zip(pt2, origin_pt)
    )
    return ((scaled_xmin, scaled_ymin), (scaled_xmax, scaled_ymax))
