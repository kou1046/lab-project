from tkinter.filedialog import askdirectory
import glob
from combiner import complementidcreator

base_dir = askdirectory(initialdir="/outputs")
ids = glob.glob(f"{base_dir}/ID/*.csv")
jpgs = glob.glob(f"{base_dir}/ID/*.jpg")
creator = complementidcreator.ComplementIdCreator(ids, jpgs, base_dir)
creator.mainloop()
