
import questionary
import glob
from mypkg.submodules_aggregator import *

choiced = questionary.select("select output",glob.glob("/outputs/*")).ask()
dir_ = FrameElementDirectory(choiced)
factory = FrameFactory(dir_)
factory.create()
print()