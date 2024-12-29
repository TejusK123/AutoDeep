from setuptools import setup, find_packages

setup(
    name="AutoDeep",
    version="0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="A brief description of the package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_package",
    packages=find_packages(),
    scripts=["scripts/AutoDeep.sh"],  # Include your .sh script
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        "console_scripts": [
            "AutoDeep=AutoDeep_wrapper:main",  # Replace with your CLI entry
        ],
    },
)
