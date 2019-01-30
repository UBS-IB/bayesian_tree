#!/usr/bin/env python

# Copyright (c) 2018-2019 UBS AG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))


with open(path.join(here, "README.md")) as l:
    long_description = l.read()


requirements = [
    'matplotlib==2.2.*',
    'scipy==1.1.*',
    'numpy==1.13.*',
    'pandas==0.23.*',
    'requests==2.19.1',
    'scikit-learn==0.19.*',
]


setup(
    name='bayesian-decision-tree',
    version='0.1.0',
    description='An implementation of the paper: A Bayesian Tree Algorithm by Nuti et al.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/UBS-IB/bayesian_tree',
    author='UBS SDL Data Science',
    author_email='dl-frc-sdl-datascience@ubs.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'License :: OSI Approved :: Apache License Version 2.0',
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=requirements,
)
