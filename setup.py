from setuptools import setup

setup(
    name='CorrGrad',
    version='0.1.0',    
    description='A library for implementing active learning using correlation of gradients',
    url='https://github.com/tirsgaard/gradient_correlation',
    author='Rasmus Tirsgaard',
    license='BSD 2-clause',
    packages=['src'],
    install_requires=[
        "torch",
        "torchvision",
        "pyyaml",
        "easydict",                   
                ],
)