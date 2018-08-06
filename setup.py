from setuptools import setup

setup(
    name='keras-bi-lm',
    version='0.0.2',
    packages=['keras_bi_lm'],
    url='https://github.com/PoWWoP/keras_bi_lm',
    license='LICENSE',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy',
        'keras',
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
