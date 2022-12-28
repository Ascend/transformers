import setuptools
import pathlib

setuptools.setup(
    name="transformers_npu",
    version="4.18.0",
    description="An adaptor for transformers v4.18.0 on Ascend NPU",
    url="https://gitee.com/ascend/transformers",
    package=['transformers_npu'],
    install_package_data=True,
    license="Apache2",
    license_file="./LICENSE",
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)