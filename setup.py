from setuptools import setup

setup(
    name='keras-bi-lm',
    version='0.21.0',
    packages=['keras_bi_lm'],
    url='https://github.com/CyberZHG/keras-bi-lm',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='Train the Bi-LM model and use it as a feature extraction method',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy',
        'keras',
    ],
    classifiers=(
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
