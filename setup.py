from setuptools import setup, find_packages

setup(
    name="datamoire",
    version="1.0.0",
    description="Python library for data processing",
    url="https://github.com/lsdrfrx/datamoire",
    author="Christian Guetnga",
    author_email="lsdrfrx@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchmetrics",
        "plotly",
        "dataclasses",
        "statsmodels",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)