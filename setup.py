from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='ModEA',
    version='0.3.1',
    description='A modular evolutionary algorithm framework, mostly tailored to a modular implementation of the CMA-ES',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Sander van Rijn',
    author_email='svr003@gmail.com',
    url="https://github.com/sjvrijn/ModEA",
    packages=['modea'],
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
