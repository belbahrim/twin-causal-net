from setuptools import setup

setup(
    name='twincausal',  # Required
    version='0.1.5',  # Required
    description='Twin causal model',  # Optional
    py_modules=["twincausal.model","twincausal.core.test", "twincausal.core.train", "twincausal.core.inference","twincausal.core.training", "twincausal.losses.losses", "twincausal.models.models", "twincausal.models.twin", "twincausal.proximal.proximal", "twincausal.utils.generator", "twincausal.utils.logger","twincausal.utils.performance", "twincausal.utils.preprocess", "twincausal.core.train_sk_twin"],
    package_dir={'': 'src'}, 
)
