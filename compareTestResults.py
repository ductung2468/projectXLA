RES_PATH = "plate_results.csv" # label data
OUT_PATH = "output.csv"

def get_data(path):
	print(f"\nReading {path}")
	data = {}
	with open(path, "r") as file:
		file.readline() # skip the first line
		while True:
			line = file.readline()
			if not line:
				break
			tmp = line.split(",")
			if len(tmp) != 2:
				raise Exception("Something went wrong")
			name = tmp[0].strip()
			values = tmp[1].strip().split("||")
			values.sort()
			values = "||".join(values).strip("|")
			data[name] = values
			print(f"{name} : {values}")
	return data

res_data = get_data(RES_PATH)
out_data = get_data(OUT_PATH)

total = len(res_data)
count = 0

for k, v in out_data.items():
	if out_data[k] == res_data[k]:
		count += 1

print(f"Number of valid: {count}/{total}")
print(f"Accuracy: {count / total * 100:.2f}%")

input("Enter to continue...")