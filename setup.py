import re

from setuptools import find_namespace_packages, setup

# Ensure we match the version set in optimum/version.py
try:
    filepath = "optimum/ascend/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)

INSTALL_REQUIRES = [
    "transformers==4.28.1",
    "datasets",
    "tokenizers",
    "sentencepiece",
    "scipy",
    "pillow",
    "evaluate",
]

TEST_REQUIRES = [
    "pytest",
    "psutil",
    "parameterized",
    "GitPython",
    "optuna",
]

QUALITY_REQUIRES = [
    "black",
    "isort",
]

EXTRA_REQUIRE = {
    "testing": TEST_REQUIRES,
    "quality": QUALITY_REQUIRES,
}

setup(
    name="optimum-ascend",
    version=__version__,
    description="Optimum Ascend is the interface between the Hugging Face Transformers library and Ascend NPU. "
                "It provides a set of tools enabling easy model loading and fine-tuning on single- and multi-NPU "
                "settings for different task.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, mixed-precision training, fine-tuning, ascend, npu",
    url="",
    author="",
    author_email="",
    license="Apache",
    packages=find_namespace_packages(include=["optimum*"]),
    install_requires=INSTALL_REQUIRES,
    extra_require=EXTRA_REQUIRE,
    include_package_data=True,
    zip_safe=False,
)