from setuptools import setup, find_packages

setup(
    name="madurify",
    version="0.1.0",
    description="Face swapping application using geometric transformations",
    author="iakzs",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "dlib>=19.24.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pillow>=10.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-multipart>=0.0.6",
    ],
    entry_points={
        "console_scripts": [
            "madurify=src.cli.main:main",
        ],
    },
    python_requires=">=3.11",
)

