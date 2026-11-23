# Start here

This folder is a complete web app, the **Voice-Enabled Data Analyst**. It runs on your own
computer and keeps your data there.

## 1. Install Python (once)

You need **Python 3.10, 3.11, 3.12 or 3.13** (3.14 does not work yet). Already have one of
these? Skip this step.

* **Windows:** download Python 3.13 from https://www.python.org/downloads/windows/, run the
  installer and tick **"Add python.exe to PATH"** on its first screen.
* **macOS:** the Python 3.13 installer from https://www.python.org/downloads/macos/, or
  `brew install python@3.13`.
* **Linux:** your distribution's Python 3.10 to 3.13 with venv, for example
  `sudo apt install python3 python3-venv`.

## 2. Start the app

* **Windows:** double-click **`run.bat`**.
* **macOS / Linux:** open a terminal in this folder and run **`./run.sh`** (or `bash run.sh`).

The first start takes a few minutes, because it downloads the packages once. It asks you two
things:

1. A **Super Admin** username, e-mail and password. This is the platform owner account; the
   password stays hidden while you type.
2. Whether to **add a demo company with sample data**. Press Enter for yes. It prints the
   passwords of two demo users (`demo.admin` and `demo.analyst`): write them down.

Then the browser opens **http://localhost:8501**. Later starts take a few seconds and ask nothing.

Sign in as `demo.admin` to try everything: upload files, check data quality, clean data and ask
questions. The demo users are asked to choose their own password at their first sign-in.

To stop the app, close its window (or press Ctrl+C in the terminal).

## Using Antigravity

Open this folder (File > Open Folder), then either:

* open a terminal (Terminal > New Terminal) and run `.\run.bat` (Windows) or `./run.sh`
  (macOS / Linux), and answer its two questions there, or
* ask the agent: **"Set up and start this app as described in AGENTS.md."** It will ask you for
  the Super Admin details.

## Put it on GitHub (optional)

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

`.gitignore` already keeps your data, database, `.env` settings and the `.venv` folder out of git.

## Good to know

* Without extra software, questions are answered by a built-in rule-based interpreter. For
  free-form questions, pick one AI option:
  * **Gemini (easiest):** get a free key at https://aistudio.google.com/apikey, then in the `.env`
    file set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=<your key>`, and restart. Each question and
    the column names go to Google (never the data itself); on the free key Google may use and read
    them, so keep confidential names out of questions. Details: README, "Cloud AI with Gemini".
  * **Ollama (private):** runs on your computer, nothing leaves it, but it needs about 6 GB of free
    memory and a 2 to 5 GB download. README, "Local AI with Ollama".
* Offline voice input and e-mail alerts are optional: README, "Voice questions" and "E-mail
  alerts".
* Something went wrong? See README, "Troubleshooting". To reinstall the packages, delete the
  `.venv` folder and start again. Deleting `data` and `storage` erases all accounts and datasets.
* Details of what was tested: `docs/VERIFICATION_REPORT.md`.
