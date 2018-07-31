"""The setup file for CatLearn."""
import setuptools


def parse_requirements(filename):
    """Load requirements from requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


install_reqs = parse_requirements('./requirements.txt')
reqs = [str(req) for req in install_reqs]

setuptools.setup(
    name="CatLearn",
    version="0.4.4.dev5",
    url="https://github.com/SUNCAT-Center/CatLearn",

    author="Paul C. Jennings",
    author_email="pcjennings@stanford.edu",

    description="Machine Learning using atomic-scale calculations.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",

    license='GPL-3.0',

    packages=setuptools.find_packages(),
    package_data={'catlearn': ['data/*.json',
                               'api/magpie/*',
                               'api/magpie/*/*',
                               'api/magpie/*/*/*',
                               ]},

    install_requires=reqs,

    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
