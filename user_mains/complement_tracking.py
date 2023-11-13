from __future__ import annotations

import sys
from pathlib import Path
from tkinter import filedialog

sys.path.append(".")

from submodules.deepsort_openpose.api.applications import ComplementTrackingApplication

base_dir = Path(filedialog.askdirectory(initialdir="./submodules/deepsort_openpose/outputs"))
csvs = base_dir.glob("ID/*.csv")
jpgs = base_dir.glob("ID/*.jpg")
app = ComplementTrackingApplication([str(csv) for csv in csvs], [str(jpg) for jpg in jpgs], base_dir)
app.mainloop()
