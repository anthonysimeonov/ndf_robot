import os

from setuptools import find_packages
from setuptools import setup

dir_path = os.path.dirname(os.path.realpath(__file__))


def read_requirements_file(filename):
    req_file_path = '%s/%s' % (dir_path, filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]


packages = find_packages('src')
# Ensure that we don't pollute the global namespace.
for p in packages:
    assert p == 'ndf_robot' or p.startswith('ndf_robot.')


def pkg_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('../..', path, filename))
    return paths


extra_pkg_files = pkg_files('src/ndf_robot/descriptions')

setup(
    name='ndf_robot',
    author='Anthony Simeonov, Yilun Du',
    license='MIT',
    packages=packages,
    package_dir={'': 'src'},
    package_data={
        'ndf_robot': extra_pkg_files,
    },
    install_requires=read_requirements_file('requirements.txt') + ['airobot @ git+https://github.com/Improbable-AI/airobot.git@qa#egg=airobot']
)

