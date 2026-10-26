import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from models import Generator

st.set_page_config(
    page_title="2D Ising Model | Conditional GAN",
    page_icon="🧲",
    layout="wide",
)

st.title("🧲 2D Ising Model | Conditional GAN")
st.write(
    "Generate 32 × 32 Ising spin configurations using a "
    "temperature-conditioned Generative Adversarial Network."
)

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "ising_gan_generator.pt"
LATENT_DIM = 100
LATTICE_SIZE = 32
DEVICE = torch.device("cpu")  # CPU is reliable on Streamlit Community Cloud


@st.cache_resource
def load_generator():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing model weights: {CHECKPOINT_PATH}. "
            "Make sure the checkpoint is in the checkpoints folder."
        )
    model = Generator().to(DEVICE)
    state_dict = torch.load(
        CHECKPOINT_PATH, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def calculate_energy(spins):
    """Energy per spin for J=1, zero external field, periodic boundaries."""
    right = np.roll(spins, -1, axis=1)
    down = np.roll(spins, -1, axis=0)
    return float(-np.mean(spins * (right + down)))


def calculate_magnetization(spins):
    return float(np.mean(spins))


def calculate_abs_magnetization(spins):
    return float(abs(np.mean(spins)))


def calculate_correlation(spins):
    right = np.roll(spins, -1, axis=1)
    return float(np.mean(spins * right))


with st.sidebar:
    st.header("Generation settings")
    temperature = st.slider(
        "Temperature", min_value=1.0, max_value=4.0,
        value=3.1, step=0.1
    )
    seed = st.number_input(
        "Random seed", min_value=0, max_value=999999,
        value=39, step=1
    )
    generate = st.button("Generate pattern", type="primary", use_container_width=True)
    st.caption("Temperature is normalized to the model's training range [-1, 1].")

if generate or "spins" not in st.session_state:
    try:
        generator = load_generator()
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        normalized_temperature = 2.0 * (float(temperature) - 1.0) / 3.0 - 1.0
        noise = torch.randn(1, LATENT_DIM, device=DEVICE)
        temp_tensor = torch.tensor(
            [[normalized_temperature]], dtype=torch.float32, device=DEVICE
        )
        with torch.no_grad():
            output = generator(noise, temp_tensor)
        continuous_pattern = output[0, 0].cpu().numpy()
        spins = np.where(continuous_pattern >= 0, 1, -1).astype(np.int8)
        st.session_state["spins"] = spins
        st.session_state["temperature"] = float(temperature)
        st.session_state["seed"] = int(seed)
        st.session_state["continuous_pattern"] = continuous_pattern
    except Exception as error:
        st.error(f"Could not load or run the Generator: {error}")
        st.info(
            "Confirm that the checkpoint is the supplied ising_gan_generator.pt "
            "and that models.py has not been changed."
        )
        st.stop()

spins = st.session_state["spins"]
col1, col2 = st.columns([1.15, 1])

with col1:
    st.subheader("Generated spin configuration")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(spins, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(f"T = {st.session_state['temperature']:.1f}")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)
    plt.close(fig)
    st.download_button(
        "Download spin grid (CSV)",
        data="".join(",".join(map(str, row)) + "\n" for row in spins),
        file_name=f"ising_spins_T{st.session_state['temperature']:.1f}.csv",
        mime="text/csv",
    )

with col2:
    st.subheader("Physical measurements")
    energy = calculate_energy(spins)
    abs_mag = calculate_abs_magnetization(spins)
    magnetization = calculate_magnetization(spins)
    correlation = calculate_correlation(spins)
    st.metric("Energy per spin", f"{energy:.4f}")
    st.metric("Absolute magnetization", f"{abs_mag:.4f}")
    st.metric("Mean magnetization", f"{magnetization:.4f}")
    st.metric("Nearest-neighbour correlation", f"{correlation:.4f}")
    st.caption(
        "Measurements are computed from the thresholded generated grid. "
        "They are not Monte Carlo reference values."
    )

with st.expander("About this model"):
    st.write(
        "The Generator takes 100 latent noise values and one normalized "
        "temperature value. Its output is a 32 × 32 continuous grid, which "
        "the app thresholds into Ising spins −1 and +1. The trained weights "
        "are loaded from the included checkpoint."
    )

st.divider()
st.caption("2D Ising Model | Conditional GAN | PyTorch + Streamlit")
