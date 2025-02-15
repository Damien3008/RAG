#!/bin/bash
gunicorn --bind 0.0.0.0:5001 --chdir backend wsgi:app --timeout 120 