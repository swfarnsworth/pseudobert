from setuptools import setup, find_packages


setup(
    name='pseudobert',
    packages=find_packages(),
    version='0.0.0',
    description='Pseudo data generation with BERT',
    author='Steele Farnsworth',
    install_requires=[
        'torch',
        'transformers',
        'scispacy',
        'spacy',
        'bratlib @ git+https://github.com/swfarnsworth/bratlib.git',
        'en_core_sci_lg @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_lg-0.2.5.tar.gz'
    ]
)
