# libyt Documents Development

- Require Python packages in `requirements.txt`.
- Enter `doc` folder, and build the doc:
  ```bash
  sphinx-build -n -W --keep-going -b html . _build/
  ```
- Use auto-build to get updates while writing the doc:
  ```bash
  sphinx-autobuild -n -W -b html . _build/ --port 8001
  ```

#### Notes
- When doing cross-referencing between different files, use the anchor generated in markdown. Don't use the anchor shown in the browser. The Anchor slug used in furo (sphinx theme) after built (in browser) and the one used in myst markdown are not the same!