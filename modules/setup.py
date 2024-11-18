from setuptools import find_packages, setup

setup(
    name='unsupervised_learning',
    version='0.1.6.3',
    description='Unsupervised learning functions',
    author='John Doe',
    author_email='jdoe@example.com',
    packages=find_packages(),  # ['unsupervised_learning'],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'numba'
    ],
)
