# Reference from PyTorch: https://github.com/pytorch/pytorch/blob/master/setup.py#L186
from __future__ import print_function
import sys
import platform


if sys.version_info < (3,):
    print("Python 2 has reached end-of-life and is no longer supported by PyTorch.")
    sys.exit(-1)
if sys.platform == 'win32' and sys.maxsize.bit_length() == 31:
    print("32-bit Windows Python runtime is not supported. Please switch to 64-bit Python.")
    sys.exit(-1)

python_min_version = (3, 6, 2)
python_min_version_str = '.'.join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print("You are using Python {}. Python >={} is required.".format(
        platform.python_version(),
        python_min_version_str
    ))
    sys.exit(-1)


# Set up
import os
from setuptools import setup, find_packages


root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(os.path.join(root, 'version.txt'), 'r') as f:
    version = f.read().strip()

setup(
    name='comvex',
    version=version,
    keywords='pytorch einops computer-vision',
    description='Implementations of Recent Papers in Computer Vision',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/blakechi/ComVEX',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    author='Blake Chi',
    author_email='blakechi.chiaohu@gmail.com',
    packages=find_packages(exclude=['examples', 'tests']),
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=['torch >= 1.8.1', 'einops >= 0.3.0', 'torchvision >= 0.8.2'],
)