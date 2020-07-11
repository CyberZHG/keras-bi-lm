#!/usr/bin/env bash
pycodestyle --max-line-length=120 keras_bi_lm tests && \
    nosetests --nocapture --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --cover-package=keras_bi_lm --with-doctest