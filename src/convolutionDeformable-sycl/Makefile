#
#==============================================================================
# Targets to Build
#===============================================================================

build: setup.py
	python3 setup.py build develop

clean:
	rm -rf *.egg-info *.so build __pycache__

run: build
	$(LAUNCHER) python3 main.py # warmup run
	$(LAUNCHER) python3 main.py
