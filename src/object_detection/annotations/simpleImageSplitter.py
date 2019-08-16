import cv2
import os
import glob

INFILEEXT  = ".tif"
OUTFILEEXT = INFILEEXT#".png"

CWD = os.getcwd()

INFOLDER  = os.path.join(CWD, "img")

HEIGHT = 500
WIDTH  = 4000

if HEIGHT > WIDTH: # TODO: this is not implemented (yet?) because for "annotation" dataset width > height
    NUMPATCHES = HEIGHT // WIDTH
    NEWHEIGHT  = WIDTH
    NEWWIDTH   = WIDTH
    INDEXARR   = list(range(0,HEIGHT+1,WIDTH))
elif WIDTH > HEIGHT:
    NUMPATCHES = WIDTH // HEIGHT
    NEWHEIGHT  = HEIGHT
    NEWWIDTH   = HEIGHT
    INDEXARR   = list(range(0,WIDTH+1,HEIGHT))
else:
    NUMPATCHES = 1
    NEWHEIGHT  = HEIGHT
    NEWWIDTH   = WIDTH
    INDEXARR   = []
    print(f"WARNING: Height is the same as width ({NEWHEIGHT}).")
    # Could quit here..? 

PATCHCOUNTARR = list(range(NUMPATCHES))

print(f"New img: {NUMPATCHES} patches at dims. {NEWWIDTH}x{NEWHEIGHT}")
print(INDEXARR)
print(PATCHCOUNTARR)

OUTFOLDER = os.path.join(CWD, f"img_patches_{NUMPATCHES}") 

if not os.path.exists(OUTFOLDER):
    os.makedirs(OUTFOLDER)
    print("WARNING: Making folder, {}".format(OUTFOLDER))

infiles = glob.glob(INFOLDER + '/*' + INFILEEXT)

# TODO: Configure this with more than just width > height mode?
for file_ in infiles:
    print(f"***** PROCESSING {file_} *****")
    frame = cv2.imread(file_)
    if frame is None:
        print("WARNING: {} cannot be read. Skipping...".format(file_))
        continue
    
    basename = os.path.splitext(os.path.basename(file_))[0]
    for i in PATCHCOUNTARR:
        leftIdx  = INDEXARR[i]
        rightIdx = INDEXARR[i+1]
        outFilename = f"{basename}_{i}{OUTFILEEXT}"
        cv2.imwrite(os.path.join(OUTFOLDER, outFilename), frame[:,leftIdx:rightIdx,:])

# EOF