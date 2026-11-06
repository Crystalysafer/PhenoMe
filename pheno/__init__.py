# flake8: noqa

import logging


__appname__ = 'PhenoMe'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__appname__)


from pheno._version import __version__

