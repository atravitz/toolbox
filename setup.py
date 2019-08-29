import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="toolbox",
    version="0.0.1",
    author="Alyssa Travitz",
    author_email="atravitz@umich.edu",
    description="helpful tools for research",
    long_description = long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/atravitz/toolbox.git",
    packages=setuptools.find_packages(exclude=('tests')),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License : OSI Approved :: MIT License",
    ]
)
