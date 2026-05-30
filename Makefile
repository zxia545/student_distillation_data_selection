.PHONY: check smoke test

check: test
	find examples scripts -name '*.sh' -print0 | xargs -0 -r -n1 bash -n
	python -m compileall -q scas tests
	git diff --check

smoke:
	bash examples/run_demo.sh

test:
	python -m unittest discover -s tests
