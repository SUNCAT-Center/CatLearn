"""https://dzone.com/articles/python-script-delete-merged"""
from subprocess import check_output
import sys

msg = '*****************************************************************\n'
msg += 'Did not actually delete anything yet, pass in --confirm to delete\n'
msg += '*****************************************************************'


def get_merged_branches():
    """A list of merged branches, not couting the current branch or master."""
    raw_results = check_output(
        'git branch --merged upstream/master', shell=True)

    return [b.strip() for b in raw_results.split('\n'.encode())
            if b.strip() and not b.startswith(
                '*'.encode()) and b.strip() != 'master'.encode()]


def delete_branch(branch):
    branch = ''.join(chr(x) for x in branch)
    return check_output('git branch -D %s' % branch, shell=True)


if __name__ == '__main__':
    dry_run = '--confirm' not in sys.argv
    for branch in get_merged_branches():
        if dry_run:
            print(branch)
        else:
            print(delete_branch(branch))
    if dry_run:
        print(msg)
