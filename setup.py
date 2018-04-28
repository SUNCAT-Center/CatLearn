"""The setup file for CatLearn."""
import setuptools

setuptools.setup(
    name="CatLearn",
    version="0.4.1.post1",
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

    install_requires=['ase==3.16.0',
                      'h5py==2.7.1',
                      'networkx==2.1.0',
                      'numpy==1.14.2',
                      'pandas==0.22.0',
                      'pytest-cov==2.5.1',
                      'scikit-learn==0.19.1',
                      'scipy==1.0.1',
                      'tqdm==4.20.0',
                      ],

    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
