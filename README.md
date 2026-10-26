# 2D Ising Model Conditional GAN — Streamlit app

## Files
- `app.py`: Streamlit interface and physical measurements
- `models.py`: Generator architecture matching the supplied checkpoint
- `checkpoints/ising_gan_generator.pt`: trained Generator weights
- `requirements.txt`: Python dependencies

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload these files and the `checkpoints` folder.
2. Open https://share.streamlit.io/ and choose **Create app**.
3. Select the repository, branch, and `app.py` as the main file.
4. Deploy.

The app uses CPU inference for compatibility with Streamlit Community Cloud.
Keep the checkpoint file in the repository if its size is within GitHub's file limits.
Do not commit API keys or other secrets.
