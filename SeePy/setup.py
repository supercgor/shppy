from setuptools import setup, find_packages

setup(
    name='seepy',
    version='0.0.1',
    description=(
        "Season's Python package for Physics"
    ),
    long_description=open('README.md').read(),
    author='Chon-Hei Lo',
    author_email='siumabon123@gmail.com',
    maintainer='Chon-Hei Lo',
    maintainer_email='siumabon123@gmail.com',
    license='MIT License',
    platforms=["linux", "macos"],
    url='https://github.com/supercgor/seepy',
    packages=find_packages(),
    # entry_points={
    #     'console_scripts': [
    #         'seepy = seepy.main:main',
    #     ]
    # },
    # package_data={
    #     'seepy': ['testdata/*']
    # },
    classifiers=[
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    python_requires='>=3.9'
)