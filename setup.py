from setuptools import setup, find_packages

setup(
    name='evolvingsystems',
    version='0.1',
    url='https://github.com/AndressaStefany/evolving_systems',
    author='Andressa Stéfany S de Oliveira',
    author_email='astefanysoliveira@gmail.com',
    description='Evolving Systems',
    packages=find_packages("Algoritms/"),
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)
