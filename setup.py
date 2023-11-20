from setuptools import find_packages, setup

setup(
    name="rt1-pytorch",
    packages=find_packages(exclude=[]),
    version="0.1.0",
    license="MIT",
    description="PyTorch implementation of the RT-1.",
    author="Rohan Potdar",
    author_email="rohanpotdar138@gmail.com",
    long_description_content_type="text/markdown",
    url="https://github.com/Rohan138/rt1-pytorch",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "transformers",
        "attention mechanism",
        "robotics",
    ],
    install_requires=[
        "torch>=1.9",
        "scikit-image",
        "sentence-transformers",
        "tensorflow",
        "tensorflow_datasets",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
