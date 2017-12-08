"""The pip setup file for AtoML."""
import setuptools


def parse_requirements(filename):
    """Load requirements from requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


install_reqs = parse_requirements('./requirements.txt')
reqs = [str(req) for req in install_reqs]


setuptools.setup(
    name="AtoML",
    version="0.1.0",
    url="https://gitlab.com/atoML/AtoML",

    author="Paul C. Jennings, Martin H. Hansen",
    author_email="pcjennings@stanford.edu",

    description="Machine Learning using atomic-scale calculations.",
    long_description=open('README.md').read(),

    packages=setuptools.find_packages(),

    install_requires=reqs,

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
