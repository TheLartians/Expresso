from setuptools import setup, Extension, find_packages
from glob import glob

setup(
    name='expresso',
    version='0.1',
    description='A symbolic expression manipulation library.',
    license='MIT',
    author='Lars Melchior',
    author_email='thelartians@gmail.com',

    url='https://github.com/TheLartians/Expresso',
    
    packages=find_packages(exclude=['tests*']),
    
    keywords='expression symbolic manipulation algebra',

    extras_require={
        'pycas':['numpy','mpmath']
    },

    zip_safe=False,

    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7'
    ],

    ext_modules=[
        Extension('_expresso',
                  sources = glob('source/expresso/*.cpp') + ['libs/sha256/sha256.cpp','source/python.cpp'],
                  include_dirs=['libs'], # assuming your project include files are there
                  libraries=['boost_python'], # those are the linked libs
                  library_dirs=['/'],
                  extra_compile_args=['-g','-std=c++11','-Wno-unknown-pragmas','-O3'] # some other compile args
                  ),
        ]
)
