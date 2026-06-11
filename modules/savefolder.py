from pathlib import Path



def create_savefolder():
    make_checkpoint_folder()
    make_mpkg_folder()
    return


def make_checkpoint_folder():
    checkpoint_folder_path = Path("savefolder/checkpoint")
    checkpoint_folder_path.mkdir(parents=True, exist_ok=True)
    return


def make_mpkg_folder():
    mpkg_folder_path = Path("savefolder/mpkg/tmp")
    mpkg_folder_path.mkdir(parents=True, exist_ok=True)
    return