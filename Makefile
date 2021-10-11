#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_VERSION = python3.8
VIRTUALENV := build/virtualenv
SHELL := /bin/bash

#################################################################################
# COMMANDS                                                                      #
#################################################################################

# Set the default location for the virtualenv to be stored
# Create the virtualenv by installing the requirements and test requirements

VIRTUALENV := build/virtualenv

$(VIRTUALENV)/.installed:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python python3 $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r requirements.txt
	$(VIRTUALENV)/bin/python3 setup.py develop --no-deps
	touch $@

# Update the requirements to latest. This is required because typically we won't
# want to incldue test requirements in the requirements of the application, and
# because it makes life much easier when we want to keep our dependencies up to
# date.

.PHONY: update-requirements-txt
update-requirements-txt: VIRTUALENV := /tmp/update-requirements-virtualenv
update-requirements-txt:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python $(PYTHON_VERSION) $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r unpinned_requirements.txt
	echo "# Created by 'make update-requirements-txt'. DO NOT EDIT!" > requirements.txt
	$(VIRTUALENV)/bin/pip freeze | grep -v pkg-resources==0.0.0 >> requirements.txt

.PHONY: virtualenv
virtualenv: $(VIRTUALENV)/.installed

# Delete all compiled Python files
.PHONY: clean
clean:
	$(VIRTUALENV)/bin/python setup.py clean --all
	-find . -not -path ./build -type f -name "*.py[co]" -print -delete
	-find .  -not -path ./build -type d -name "__pycache__" -print -exec rm -rv {} \;
	-find .  -not -path ./build -type d -name ".pytest_cache" -print -exec rm -rv {} \;
	-find .  -not -path ./build -type d -name "*.egg-info" -print -exec rm -rv {} \;
	-find .  -not -path ./build -type d -name "*.tox" -print -exec rm -rv {} \;
	-find .  -not -path ./build -type d -name "mantis-*" -delete
	-find .  -not -path ./build -type d -name "dist" -delete
