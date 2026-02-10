try:
    import mamba_ssm
    print('Mamba-SSM: INSTALLED -', mamba_ssm.__version__)
except ImportError:
    print('Mamba-SSM: NOT INSTALLED')
