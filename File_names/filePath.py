import os
class FilePath:

    def filePaths(self,folder_path):
        filenames = []
        for file in os.listdir(folder_path):
            filenames.append(os.path.join('images',file)) # getting path of all images
        return filenames  # returning filenames with path
