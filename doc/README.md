# libyt Documents Development

- Require Python packages in `requirements.txt`.
- Build the doc:
  ```bash
  sphinx-build -n -W --keep-going -b html doc/ doc/_build/
  ```
- Use auto-build to get updates while writing the doc:
  ```bash
  sphinx-autobuild -n -W -b html doc/ doc/_build/ --port 8001
  ```
