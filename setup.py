from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

extras_require = {
    "dev": [
        "torchtestcase",
        "pytest",
        "pytest-cov",
        "ruff",
        "parameterized",
    ],
    "examples": ["matplotlib", "scikit-learn", "pandas"],
}

extras_require["all"] = sum(extras_require.values(), [])


setup(
    name="flowcon",
    version="1.0.0",
    description="Normalizing flows in PyTorch. An extension of nflows.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FabricioArendTorres/FlowConductor/",
    author="Fabricio Arend Torres, Marcello Massimo Negri, Jonathan Aellen",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    install_requires=[
        "numpy",
        "torch",
        "ninja",
        "h5py",
    ],
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha ",
    ],
    dependency_links=[],
)
