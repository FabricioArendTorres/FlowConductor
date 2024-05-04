from os import path
from setuptools import find_packages, setup


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
setup(
    name="flowcon",
    version="0.0.4",
    description="Normalizing flows in PyTorch. An extension of nflows.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/FabricioArendTorres/FlowConductor/",
    author="Fabricio Arend Torres, Marcello Massimo Negri, Jonathan Aellen",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    # setup_requires=['pbr'],
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "torch",
        "torchdiffeq",
        "umnn",
        "tqdm",
        "ninja",
        "scikit-learn",
        "h5py",
        "torchtestcase",
        "parameterized",
        # testing
        "flake8",
        "pytest",
        "pytest-cov",
        "black"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha "
    ],
    # python_requires='>3.7',
    dependency_links=[],
)

