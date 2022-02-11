import io
import os

from setuptools import setup, find_packages

dir = os.path.dirname(__file__)

with io.open(os.path.join(dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='project-d',
    version='1.0',
    description='Tool to share',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/private_directory',
    author='Me',
    author_email='my_email@place.org',
    license='GNU',
    install_requires=['pandas>=0.25', 'lxml','matplotlib'],
    python_requires='>=3',
    packages=find_packages()
)

//https://medium.com/analytics-vidhya/how-to-create-a-python-library-7d5aea80cc3f
