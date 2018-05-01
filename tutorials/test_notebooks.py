"""Tests for the jupyter notebooks."""
import os
import subprocess
from sty import fg, ef, rs


def test_notebooks(notebook):
    """Function to run a notebook.

    Parameters
    ----------
    notebook : str
        Path and name of notebook to be tested.
    """
    # Break up the path.
    psplit = notebook.split('/')
    name = psplit.pop(-1)
    print(fg.red + '\ntesting notebook: ' + fg.rs + fg.green +
          '{}\n'.format(name) + fg.rs)
    dirpath = '/'.join(psplit)

    # Execute the jupyter notebook.
    cmd = 'jupyter nbconvert {} --to notebook --execute '.format(notebook)
    cmd += '--ExecutePreprocessor.timeout=6000 --output out_file'
    subprocess.call(cmd, shell=True)

    # Try and clean up output. Otherwise fail and exit.
    try:
        os.remove('{}/out_file.ipynb'.format(dirpath))
    except FileNotFoundError:
        print(ef.bold + fg(201) + '\n{} failed\n'.format(name) + fg.rs +
              rs.bold)
        exit()


if __name__ == '__main__':
    path = os.getcwd()

    flist = []

    # Find all notebooks in the tutorials directory.
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'checkpoint' not in file:
                if file.endswith(".ipynb"):
                    flist.append(os.path.join(root, file))

    # Run all the tests.
    for f in flist:
        test_notebooks(f)
