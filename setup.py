from setuptools import setup, find_packages

setup(
    name="fitness-leader-election",
    version="1.0.0",
    description="Fitness-Based Leader Election for Autonomous Drone Swarms",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "networkx>=3.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "tqdm>=4.65",
    ],
)
