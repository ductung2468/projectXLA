import os
from lpAppModel import LPAppModel

DATA_PATH = "C:\\Users\\Tung\\Desktop\\sumdoc"
RESULT_PATH = "C:\\Users\\Tung\\Desktop\\output.csv"

core = LPAppModel()

# limit = 100
# i = 1

entries = sorted(os.scandir(DATA_PATH), key=lambda e: e.name)

with open(RESULT_PATH, "w") as file:
	file.writelines("Filename,Biensoxe\n")

	for e in entries:
		if e.is_file():
			img_name = e.name

			core.detect_n_read(e.path)
			res = "||".join(core.lp_texts)

			file.writelines(f"{img_name},{res}\n")

			print(f"{img_name},{res}")

			# i += 1
			# if i > limit:
			# 	break
