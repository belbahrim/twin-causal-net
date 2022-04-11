from setuptools import setup

setup(
    name='twincausal',  # Required
    version='0.1.7',  # Required
    description='Twin causal model',  # Optional
    py_modules=["twincausal.model","twincausal.core.test", "twincausal.losses.losses", "twincausal.proximal.proximal", "twincausal.utils.data", "twincausal.utils.generator", "twincausal.utils.logger", "twincausal.utils.performance", "twincausal.utils.preprocess"],
    package_dir={'': 'src'}, 
)
