import setuptools
from distutils.core import setup


def setup_package():
    with open("README.md", "r") as f:
        readme = f.read()

    setup(
        name="cvat_reader",
        version="0.2.0",
        author="Koen Vossen",
        author_email="info@koenvossen.nl",
        url="https://github.com/eyedl/cvat_reader",
        packages=setuptools.find_packages(),
        license="BSD",
        description="Read cvat training set",
        long_description=readme,
        long_description_content_type="text/markdown",
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "License :: OSI Approved",
            "Topic :: Scientific/Engineering",
        ],
        install_requires=[
            'dataclasses;python_version<"3.7"',
        ],
    )


if __name__ == "__main__":
    setup_package()
