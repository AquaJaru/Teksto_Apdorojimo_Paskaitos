# Paleidimas

Išarchyvuokite VISĄ ZIP, VS Code atverkite katalogą ir Lab04_LT.ipynb. Select Kernel pasirinkite Python su torch, numpy, sklearn ir matplotlib. Notebookas nėra paruoštas sprendimas: pirmiausia užbaikite student_model.py ir student_search.py. NotImplementedError iki tol yra tyčinis. Visų trijų temų duomenys pakete; pasirinkite vieną. Parengtų GPT/LLM svorių nėra.

## Separate environment
```sh
python -m venv .venv
# Windows:
.venv\Scripts\python -m pip install -r requirements.txt
# macOS/Linux:
.venv/bin/python -m pip install -r requirements.txt
```
Choose this .venv in VS Code. GPU and API keys are not needed. Installation needs internet; running the prepared work does not.

```sh
python verify_attention.py
python verify_search.py
python core.py --data data/software --out outputs/software_LT --lang LT --minutes 30
python app.py --topic software --checkpoint outputs/software_LT/best.pt
```
Open the printed localhost URL. Stop the server with Ctrl+C. For research add --dim 96 --heads 4 to the training command. Check assignment.md for requirements and defence. Do not install packages into a system-wide environment.
