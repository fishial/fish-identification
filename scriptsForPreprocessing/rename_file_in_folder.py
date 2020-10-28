import glob, os
import uuid
import shutil

def rename(dir, pattern):
    folders = [x[0] for x in os.walk(dir)]
    for folder in folders:
        for x, pathAndFilename in enumerate(glob.iglob(os.path.join(folder, pattern))):
            title, ext = os.path.splitext(os.path.basename(pathAndFilename))
            file_name = str(uuid.uuid4()) + ext
            src = os.path.join(folder, file_name)
            dst = os.path.join(dir, file_name)
            print(src)
            print(dst)
            os.rename(pathAndFilename, src)
            shutil.move(src, dst)
rename(r'resources/new_pars', r'*.jpg')