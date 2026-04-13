from setuptools import setup, find_packages

setup(
    name="neuralce",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "jax",
        "flax",
        "optax",
        "numpy",
        "scipy",
        "pandas",
        "pymatgen",
        "optuna",
        "pyyaml",
        "matplotlib",
        "scikit-learn",
    ],
    entry_points={
        "console_scripts": [
            "neuralce-ablation          = neuralce.training.ablation:main",
            "neuralce-ablation-comp     = neuralce.training.ablation_comp:main",
            "neuralce-retrain           = neuralce.training.retrain:main",
            "neuralce-retrain-per-comp  = neuralce.training.retrain_per_comp:main",
            "neuralce-mcmc              = neuralce.mcmc.pt_mcmc:run_pt_mcmc",
            "neuralce-analyze-cutoffs   = neuralce.analysis.analyze_cutoffs:main",
            "neuralce-convert-dataset   = neuralce.data.convert_dataset:main",
            "neuralce-convert-nipt      = neuralce.data.convert_nipt:main",
        ],
    },
)
