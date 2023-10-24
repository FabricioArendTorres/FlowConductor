from os import path
from setuptools import find_packages, setup
from enflows.version import VERSION

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="enflows",
    version=VERSION,
    description="Normalizing flows in PyTorch. An extension of nflows.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    # url="https://github.com/FabricioArendTorres/enflows",
    # download_url = 'https://github.com/bayesiains/nflows/archive/v0.14.tar.gz',
    author="Fabricio Arend Torres, Marcello Massima Negri, Jonathan Aellen",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    # setup_requires=['pbr'],
    install_requires=[
        "matplotlib",
        "numpy",
        "tensorboard",
        "torch",
        "torchdiffeq",
        "tqdm",
        "umnn",
        "ninja",
        "scikit-learn",
        "pandas<2.0",
        "h5py"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha "
    ],
    # python_requires='>3.9',
    dependency_links=[],
)
