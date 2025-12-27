"""
Setup script for PAID-SnowNet package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().split('\n')
        if line.strip() and not line.startswith('#')
    ]
else:
    requirements = [
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
    ]

setup(
    name="paid_snownet",
    version="1.0.0",
    author="PAID-SnowNet Team",
    author_email="your.email@example.com",
    description="Physics-Aware Iterative Denoising Network for Snow Scene Restoration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/paid-snownet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "isort>=5.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "paid-snownet-train=paid_snownet.scripts.train:main",
            "paid-snownet-eval=paid_snownet.scripts.evaluate:main",
            "paid-snownet-infer=paid_snownet.scripts.inference:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
