VIRTUALENV := build/virtualenv

$(VIRTUALENV)/.installed:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python python3 $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r requirements.txt
	$(VIRTUALENV)/bin/python3 setup.py develop --no-deps
	touch $@

.PHONY: virtualenv
virtualenv: $(VIRTUALENV)/.installed
