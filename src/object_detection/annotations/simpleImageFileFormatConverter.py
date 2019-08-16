import cv2
import os
import glob

INFILEEXT  = ".tif"
OUTFILEEXT = ".png"

CWD = os.getcwd()

INFOLDERDIRNAME = "img_patches_8" # originally "img"
INFOLDER  = os.path.join(CWD, INFOLDERDIRNAME) 
OUTFOLDER = os.path.join(CWD, INFOLDERDIRNAME+"_"+OUTFILEEXT[OUTFILEEXT.rfind('.')+1:])

if not os.path.exists(OUTFOLDER):
	os.makedirs(OUTFOLDER)
	print("WARNING: Making folder, {}".format(OUTFOLDER))

infiles = glob.glob(INFOLDER + '/*' + INFILEEXT)

for file_ in infiles:
	print(f"***** PROCESSING {file_} *****")
	frame = cv2.imread(file_)
	if frame is None:
		print("WARNING: {} cannot be read. Skipping...".format(file_))
		continue
	outFilename = os.path.splitext(os.path.basename(file_))[0] + OUTFILEEXT
	cv2.imwrite(os.path.join(OUTFOLDER, outFilename), frame)

# EOF