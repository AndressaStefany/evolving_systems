from setuptools import setup, find_packages

setup(
    name='evolvingsystems',
    version='0.1',
    url='https://github.com/AndressaStefany/evolving_systems',
    author='Andressa Stéfany S de Oliveira',
    author_email='astefanysoliveira@gmail.com',
    description='Evolving Systems',
    package_dir={'': 'algoritms'},
    packages=find_packages("algoritms"),
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
