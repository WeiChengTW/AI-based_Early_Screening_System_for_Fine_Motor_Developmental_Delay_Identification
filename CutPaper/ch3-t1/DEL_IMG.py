import os
import glob


def reset_result_dir():
    folders = ["extracted", "result", "Draw_circle"]

    for folder in folders:
        if os.path.exists(folder):
            if folder == "result":
                # Delete all files in all subfolders of 'result'
                for subdir, dirs, files in os.walk(folder):
                    for file in files:
                        file_path = os.path.join(subdir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                print(f"Deleted all files in all subfolders of {folder}")
            else:
                files = glob.glob(os.path.join(folder, "*"))
                for f in files:
                    if os.path.isfile(f):
                        os.remove(f)
                print(f"Deleted all files in {folder}")
        else:
            print(f"Folder {folder} does not exist.")


if __name__ == "__main__":
    # Call the function to reset the result directory
    reset_result_dir()
