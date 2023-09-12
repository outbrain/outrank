from __future__ import annotations

import os

import setuptools


def _parse_requirements(file):
    required_packages = []
    with open(
        os.path.join(os.path.dirname(__file__), file), encoding='utf-8',
    ) as req_file:
        for line in req_file:
            required_packages.append(line.strip())
    return required_packages


def _read_description():
    with open('README.md') as description:
        return description.read()


packages = [x for x in setuptools.find_packages() if x != 'test']
setuptools.setup(
    name='outrank',
    version='0.91',
    description='OutRank: Feature ranking for massive sparse data sets.',
    long_description=_read_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/outbrain/outrank',
    author='Research Infra (Outbrain); Blaz Skrlj led the development of this project',
    license='BSD',
    entry_points={'console_scripts': ['outrank = outrank.__main__:main']},
    packages=packages,
    zip_safe=True,
    include_package_data=True,
    install_requires=_parse_requirements('requirements.txt'),
)
