# Not a full GUI, so keep the root window from appearing
def select_directory(root):
    try:
        from tkinter import Tk
        from tkinter.filedialog import askdirectory
        Tk().withdraw() 
        directory = askdirectory(
            initialdir=root
        )
    except Exception as error:
        directory = f"{root}/docs"
        print(error)
    return directory


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, default=None, help='root directory')
    opt = parser.parse_args()

    root = opt.source
    path = select_directory(root)
    print(f"You Choose {path}.")