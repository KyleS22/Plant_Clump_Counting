import os
import glob
import random
import argparse

from shutil import copyfile

def moveFiles(fileList, newFolder):
    for filepath in fileList:
        smallFilename = os.path.basename(filepath)
        copyfile(filepath, os.path.join(newFolder, smallFilename))


def main():
    parser = argparse.ArgumentParser('Randomly shuffles a folder of files into train (and validate) and test datasets.')
    parser.add_argument('-i', '--input_folder', help='folder to input files', type=str, required=True)
    parser.add_argument('-o', '--output_folder', help='folder to output new folders; defaults to input folder', type=str, default=None)
    parser.add_argument('-s', '--random_seed', help='the seed for randomization', type=int, default=7742116)
    parser.add_argument('--train_percent', help='the (float) percent of files to be used for training; 1-validation%-this will be used for testing (default=0.8)', type=float, default=0.8)
    parser.add_argument('--validate_percent', help='the (float) percent of files to be used for validation; 1-train%-this will be used for testing (default=0.0)', type=float, default=0.0)
    parser.add_argument('--file_extension', help='the file extension of the files to be shffled, inclduing the \'.\' (default=.xml)', type=str, default='.xml')

    args = parser.parse_args()
    
    if args.train_percent + args.validate_percent > 1.0:
        raise ValueError("Sum of training percent and validation percent cannot exceed 1.")
    
    infolder = args.input_folder
    
    outfolder = infolder if not args.output_folder else args.output_folder
    
    random.seed(args.random_seed)

    if args.validate_percent > 0.0:
        newFolders = {"train": args.train_percent, "validate": args.validate_percent, "test": 1-args.train_percent-args.validate_percent} 
    else:
        newFolders = {"train": args.train_percent, "test": 1-args.train_percent} 

    for dir in newFolders.keys():
        output_folder = os.path.join(outfolder, dir) # create local copy in case of modification
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Warning: making folder {output_folder}")

    fileList = glob.glob(infolder + '/*' + args.file_extension)
    lenFileList = len(fileList)
    random.shuffle(fileList)

    tempValAcc = 0
    lenNewFolders = len(newFolders.keys())
    iteration = 1 # Start at 1 because we don't want an index, but to track when we are on the last key
    for dir, ratio in newFolders.items():
        tempVal = int(ratio*lenFileList)
        if iteration < lenNewFolders:
            tempList = fileList[tempValAcc:tempVal+tempValAcc]
        else:
            tempList = fileList[tempValAcc:] # Last key, just use remainder of data
        moveFiles(tempList, os.path.join(outfolder, dir))
        tempValAcc += tempVal
        iteration += 1

if __name__ == "__main__":
    main()