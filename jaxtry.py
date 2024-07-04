import jax.numpy as jnp
import jax

print(jax.devices())


import subprocess

def get_cudnn_version():
    try:
        # This command might differ depending on your system and installation
        output = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8")
        for line in output.splitlines():
            if "cuDNN Version" in line:
                return line.split(": ")[-1]
    except subprocess.CalledProcessError:
        return None

cudnn_version = get_cudnn_version()
if cudnn_version:
    print(f"Loaded CuDNN Version: {cudnn_version}")
else:
    print("Could not determine CuDNN version.")

print(jax.__version__)

x = jnp.array([1.0, 2.0, 3.0])