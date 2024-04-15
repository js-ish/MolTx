#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from setuptools import find_packages, setup

_root = os.path.abspath(os.path.dirname(__file__))

def get_version() -> str:
    version = ""
    with open(os.path.join(_root, "moltx/_version.py"), "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                version = eval(line.split("=")[-1])
                break
    return version


with open(os.path.join(_root, "README.md"), "r", encoding="utf-8") as f:
    readme = f.read()

with open(os.path.join(_root, 'requirements.txt')) as f:
    requirements = f.readlines()

setup(
    name="moltx",
    version=get_version(),
    description="Molcule Generation",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Michael Ding",
    author_email="yandy.ding@gmail.com",
    license_files=('LICENSE',),
    url="https://gitlab.ish.org.cn/aidd/molgen/moltx-task",
    packages=find_packages(exclude=["tests", "train"]),
    find_links=["https://repo.huaweicloud.com/repository/pypi/simple"],
    install_requires=requirements,
    include_package_data=True,
    package_data={'moltx': ['data/*.json', 'data/*.txt']},
    python_requires=">=3.7.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ]
)
