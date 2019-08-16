import os
import glob
import argparse

from shutil import copyfile

def moveFiles(fileList, newFolder):
    for filepath in fileList:
        smallFilename = os.path.basename(filepath)
        copyfile(filepath, os.path.join(newFolder, smallFilename))


def main():
    parser = argparse.ArgumentParser('Moves [train/test/validate] files based off the [train/test/validate] of other files (E.g., find all corresponding image files to labels that have been previously demarcated). Ideally, run this two (three) times, inputting as the input folder one of the train, (validate), and test dirs on each run.')
    parser.add_argument('-i', '--input_folder', help='folder to input files [train, test, validate]', type=str, required=True)
    parser.add_argument('-o', '--output_folder', help='folder containing files in which to find those corresponding to the files in the input folder; folder where a new directory for the corresponding files will be made, if export folder is not given', type=str, required=True)
    parser.add_argument('-e', '--export_folder', help='folder where to move images to', type=str, default=None)
    parser.add_argument('--input_file_extension', help='the file extension of the files in the input folder, inclduing the \'.\' (default=.xml)', type=str, default='.xml')
    parser.add_argument('--output_file_extension', help='the file extension of the files in the output folder, inclduing the \'.\' (default=.png)', type=str, default='.png')

    args = parser.parse_args()
    infolder  = args.input_folder
    outfolder = args.output_folder
    exportfolder = args.export_folder
    
    dirName = os.path.basename(os.path.normpath(infolder))

    output_folder = os.path.join(outfolder if not exportfolder else exportfolder, dirName) # create local copy in case of modification
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Warning: making folder {output_folder}")

    infiles  = glob.glob(infolder + '/*' + args.input_file_extension)
    outfiles = []
    
    for filename in infiles:
        print(f"***** Processing {filename}... *****")
    
        smallFilename = (os.path.basename(filename))
        lastDotInFilename = smallFilename.rfind('.')
        filenameBasename = smallFilename[:lastDotInFilename]
        
        outfilename = filenameBasename + args.output_file_extension
        outfilepath = os.path.join(outfolder, outfilename)
        
        if os.path.isfile(outfilepath):
            outfiles.append(outfilepath)
        else:
            print(f"Warning: {outfilename} does not exist in {outfolder}.")
        
    moveFiles(outfiles, output_folder)

if __name__ == "__main__":
    main()