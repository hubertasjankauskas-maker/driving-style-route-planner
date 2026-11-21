Plaintext
driving-style-route-planner/
├── README.md
├── requirements.txt
├── data/
│   └── driving_telemetry.csv      # Sugeneruojama paleidus generate_data.py
├── results/
│   ├── metrics_summary.json       # Modelių palyginimo metrikos
│   └── model_comparison.png       # Konfuzijos matricos ir XAI grafikai
└── code/
    ├── generate_data.py           # Sintetinių telemetrijos duomenų generatorius
    ├── model.py                   # Baseline vs. Proposed modelių mokymas ir XAI
    └── routing.py                 # Adaptyvus Dijkstra maršruto planuoklis# driving-style-route-planner
