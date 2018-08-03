#!/usr/bin/env bash
rm README.rst && m2r README.md
rm dist/*
python setup.py sdist bdist_wheel
twine upload dist/*
