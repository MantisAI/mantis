__name__ = "oms"
__version__ = "0.1.0"
__description__ = """Mantis CLI and Library for NLP utils"""
__url__ = "https://github.com/MantisAI/mantis/tree/master"
__author__ = "MantisAI"
__author_email__ = "hi@mantisnlp.com"
__license__ = ""

#
# Get the API version
#

try:
    with open(".api_version", "r") as f:
        __api_version__ = f.read().strip("\n")
except Exception:
    __api_version__ = None
