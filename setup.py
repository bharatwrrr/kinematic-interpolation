from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding='utf-8') if (HERE / "README.md").exists() else "Kinematic interpolation package"

setup(
    name='kinematic_interpolation',
    version='0.1.0',
    description='A package for kinematic interpolation and related utilities',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Bharat Jayaprakash',
    url='https://github.com/bharatwrrr/kinematic-interpolation',
    license='MIT',
    packages=find_packages(exclude=("tests", "demos")),
    include_package_data=True,
    install_requires=[
        # Minimal runtime dependencies (pin/lock handled by requirements.txt for development)
        'numpy>=1.23',
        'pandas>=2.0',
        'matplotlib>=3.0',
        'pyproj>=3.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
)
