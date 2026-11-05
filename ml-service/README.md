\# SafeCrowdAI ML Service



FastAPI-based ML inference service for SafeCrowdAI.



\## Current Model



The service currently exposes CSRNet for crowd-density estimation.



CSRNet:

\- Takes an input image.

\- Generates a density map.

\- Estimates the crowd count by summing the predicted density map.



\## Project Structure



```text

ml-service/

â”œâ”€â”€ app/

â”‚   â”œâ”€â”€ main.py

â”‚   â”œâ”€â”€ schemas.py

â”‚   â”œâ”€â”€ models/

â”‚   â”‚   â”œâ”€â”€ csrnet\_wrapper.py

â”‚   â”‚   â””â”€â”€ architectures/

â”‚   â”‚       â””â”€â”€ csrnet\_arch.py

â”‚   â””â”€â”€ utils/

â”‚       â””â”€â”€ preprocessing.py

â”œâ”€â”€ checkpoints/

â”‚   â””â”€â”€ csrnet\_best.pth

â”œâ”€â”€ requirements.txt

â””â”€â”€ README.md
