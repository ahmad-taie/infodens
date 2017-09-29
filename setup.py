""" Infodens project for easy and fast NLP ML
Find at:
https://github.com/rrubino/B6-SFB1102
"""


from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sample',

    version='0.0.1',

    description='NLP features extraction and classification made simple.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/rrubino/B6-SFB1102',

    # Author details
    author='Ahmad Taie',
    author_email='tofill',

    license='GNU General Public License',


    classifiers=[

        'Development Status :: 3 - Alpha',

        'Intended Audience :: Language Researchers',
        'Topic :: NLP :: ML Tool',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License',

        # Python versions supported here
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        #'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.3',
        #'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='nlp natural language processing machine learning',

    # The project's packages via find_packages().
    packages=find_packages(),


    # Run-time dependencies to be installed by pip
    install_requires=['nltk', 'scipy','sklearn','numpy'],

    # Additional groups of dependencies (e.g. development
    # dependencies)
    extras_require={
        'dev': ['gensim'],
        'test': ['coverage'],
    },


    # executable scripts, using entry points in preference to the
    # "scripts" keyword. 
    #entry_points={
    #    'console_scripts': [
    #        'sample=sample:main',
    #    ],
    #},
)
