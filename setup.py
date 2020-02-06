from setuptools import setup, find_packages
#import versioneer


def readme():
    with open('README.rst') as f:
        import re
        long_desc = f.read()
        # strip out the raw html images
        long_desc = re.sub('\.\. raw::[\S\s]*?>\n\n', "", long_desc)
        return long_desc


setup(
    name='quchem',
    description='Quantum Chemistry Library.',
    long_description=readme(),
    url='http://quimb.readthedocs.io',
    version= '0.1.0',#versioneer.get_version(),
    #cmdclass=versioneer.get_cmdclass(),
    author='Alexis Ralli',
    author_email="alexis.ralli.18@ucl.ac.uk",
    license='MIT',
    packages=find_packages(exclude=['tests*', 'PSI4']), #finds quchem!
    install_requires=[
        'numpy>=1.12',
        'scipy>=1.0.0',
        'tqdm>=4',
        'cirq>=0.4.0',
        'openfermion>=0.9.0',
        'openfermionpsi4>=0.4',
        'networkx>=2.3'
        'openfermionpsi4>=0.4',
    ],
    extras_require={
        'optimizers': [
            'tensorflow>=1.14.0',
            'scipy>=1.0.0',
        ],
        'PSI4': [
            'psi4>=1.3.2',
        ],
        'tests': [
            'coverage',
            'pytest',
            'pytest-cov',
        ],
        'docs': [
            'sphinx',
            'ipython',
        ],
    },
    #scripts=['location_of_scripts'], #route relative to setup.py
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 1 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: Linux',
    ],
)
