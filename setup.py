from setuptools import setup, find_packages
# from Cython.Build import cythonize
import numpy as np

descrip='''
====// --------- \\\\====
Author:
Lucas Siemons
====// --------- \\\\====
'''

setup(
    name='spinSimulations',
    version='0.1',
    author='L. Siemons',
    author_email='lucas.siemons@googlemail.com',
    packages=find_packages(),
    #license='LICENSE.txt',
    package_data={'spinSimulations': ['dat/*dat'],  },
    include_package_data=True,
    description=descrip,
    include_dirs=[np.get_include()],
    zip_safe=False
    #long_description=open('README.md').read(),
    #install_requires=['numpy','scipy>=0.17.0', 'matplotlib'],
)