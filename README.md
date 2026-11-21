Markdown
# Maršruto planavimas pagal vairavimo stilių

Dirbtiniu intelektu grįsta sistema, kuri pagal telemetrijos (jutiklių) duomenis nustato vairavimo stilių (*Atsargus*, *Normalus*, *Agresyvus*) ir parinko individualizuotą maršrutą kelių grafe (*Dijkstra* algoritmu).

## Diegimas
```bash
pip install -r requirements.txt
Bash
python code/generate_data.py
python code/model.py
python code/routing.py
