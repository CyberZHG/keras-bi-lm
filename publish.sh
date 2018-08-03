#!/usr/bin/env bash
rm RAEDME.rst && m2r RAEDME.md
rm dist/*
python setup.py sdist bdist_wheel
twine upload dist/*
