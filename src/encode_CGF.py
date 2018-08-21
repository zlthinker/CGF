import sys, os.path
from subprocess import call
from multiprocessing import Pool
import timeit

def run(args):
	pcd_file = args[0]
	kpt_file = args[1]
	output_folder = args[2]
	basename = os.path.split(pcd_file)[-1]
	basename = os.path.splitext(basename)[0]
	basename = os.path.join(output_folder, basename)

	time1 = timeit.default_timer()
	lzf_file = basename+'.lzf'
	if not os.path.isfile(lzf_file):
		main_cmd = ['/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/output/projects/CGF/src/main', '-d 10 -s 1.7 -l 0.2 -t 16 --relative', 
			'-kpt', kpt_file, '-o', lzf_file, pcd_file]
		print main_cmd
		call(main_cmd)
	#return

	time2 = timeit.default_timer()
	compress_file = basename + '.npz'
	if not os.path.isfile(compress_file):
		compress_cmd = ['python', '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/output/projects/CGF/src/compress.py', lzf_file, '2244', basename]
		call(compress_cmd)
	
	time3 = timeit.default_timer()
	embed_file = basename + '_desc.npz'
	if not os.path.isfile(embed_file):
		embed_cmd = ['python', '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/output/projects/CGF/src/embedding.py', '--evaluate=True',
			'--checkpoint_model=/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/output/projects/CGF/models/embed_model_10500000.ckpt',
			"--output_file="+basename+'_desc', compress_file]
		call(embed_cmd)	

	time4 = timeit.default_timer()
	print "Time to compute spherical histogram: ", time2 - time1
	print "Time to compress: ", time3 - time2
	print "Time to embed: ", time4 - time3
	print "Time in total: ", time4 - time1

	desc_file = basename + '_desc.txt'
	if not os.path.isfile(desc_file):
		readDesc_cmd = ['python', '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/output/projects/CGF/src/readDesc.py', embed_file, desc_file]
		call(readDesc_cmd)

PCD_LIST=sys.argv[1]
KPT_LIST=sys.argv[2]	# keypoint
OUT_FOLDER=sys.argv[3]

with open(PCD_LIST) as f_PCD:
	pcd_files = f_PCD.readlines()

with open(KPT_LIST) as f_KPT:
	kpt_files = f_KPT.readlines()

print "#pcd files: ", len(pcd_files)
print "#kpt files: ", len(kpt_files)

arguments = []
for fidx in range(len(pcd_files)):
	pcd_file = pcd_files[fidx].strip()
	kpt_file = kpt_files[fidx].strip()
	args = [pcd_file, kpt_file, OUT_FOLDER]
	arguments.append(args)

process = Pool(1)
process.map(run, arguments)
