# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['neutone_sdk']

package_data = \
{'': ['*'], 'neutone_sdk': ['assets/*']}

install_requires = \
['jsonschema>=4.4.0,<5.0.0',
 'numpy==1.22.3',
 'torch==1.11.0',
 'torchaudio==0.11.0']

setup_kwargs = {
    'name': 'neutone-sdk',
    'version': '0.1.0',
    'description': 'SDK for wrapping deep learning models for deployment in Neutone',
    'long_description': '# neutone-sdk\nSDK for wrapping deep learning models for deployment in Neutone\n',
    'author': 'Qosmo',
    'author_email': 'info@qosmo.jp',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/QosmoInc/neutone-sdk.git',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
}


setup(**setup_kwargs)
