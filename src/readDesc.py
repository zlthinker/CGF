import  sys
import numpy as np

input_np = sys.argv[1]
out_path = sys.argv[2]

with np.load(input_np) as fin:
	with open(out_path, 'w') as fout:
		data = fin['data']
		number = data.shape[0]
		dim = data.shape[1]
		for n in range(number):
			for d in range(dim):
				fout.write(str(data[n][d]) + ' ')
			fout.write('\n')