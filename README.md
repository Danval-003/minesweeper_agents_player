# Minesweeper RL (JAX, GPU, batched) — README

> Entorno de *Buscaminas* para Aprendizaje por Refuerzo, 100% JAX y listo para GPU/WSL. Incluye API vectorizada, wrappers `jit`, render a video, UI interactiva con Streamlit y *stress bench*.

---

## 🚀 Características

* **Todo en JAX**: `reset`, `step`, `observe` y `action_mask` son *jit/vmap-friendly*.
* **Batches masivos**: la primera dimensión es el batch; pensado para miles de tableros en paralelo.
* **GPU-ready**: funciona en CUDA/WSL2 sin copiar arrays al host.
* **Observación rica**:

  * 9 canales one-hot (números 0..8) visibles
  * máscara de no reveladas y de banderas
  * **contexto local** opcional (radio `R`): *feature maps* “shifted” de vecinos + resúmenes por ventana.
* **Lógica de revelar**: *flood-fill* de ceros + borde numérico (8-vecinos).
* **Grabación**: script para generar **video MP4** tipo buscaminas (números, minas ocultas hasta explotar).
* **UI**: app **Streamlit** interactiva para explorar el ambiente y “ver lo que ve” el agente.
* **Bench**: *stress tests* de rendimiento (scan/unroll) en CPU/GPU.

---

## 📦 Estructura del repo

```
minesweeper_env_rl/
├── env/
│   └── __init__.py        # Ambiente JAX + wrappers jit + pytree
├── record.py              # Render a video (Pillow + imageio-ffmpeg)
├── app_streamlit.py       # UI interactiva para jugar/inspeccionar
├── stress_bench.py        # Benchmarks masivos
├── requirements.txt
└── README.md
```

---

## ✅ Requisitos

* Python 3.10–3.12
* JAX + jaxlib **con CUDA** (si usarás GPU). Ver sección de instalación.
* WSL2 (opcional, recomendado en Windows).
  **Guarda el proyecto dentro del filesystem de WSL** (por ejemplo `~/proyectos/...`) para el mejor rendimiento; evitar trabajar directamente en `/mnt/c/...` por latencias de I/O entre sistemas. ([Microsoft Learn][1])

---

## 🔧 Instalación (CPU/GPU, WSL2)

1. Crea un entorno virtual **dentro de WSL**:

```bash
cd ~/proyectos
git clone <este-repo> minesweeper_env_rl
cd minesweeper_env_rl
python3 -m venv .venv
source .venv/bin/activate
```

2. Instala dependencias del repo:

```bash
pip install -r requirements.txt
```

Si te aparece `externally-managed-environment`, crea siempre un **venv** (como arriba) o usa el *flag* de override bajo tu responsabilidad. Esto se debe al comportamiento definido en **PEP 668** que aplican distros como Debian/Ubuntu. ([TurnKey Linux][2])

3. **JAX con CUDA** (elige según tu driver):

* CUDA 12.x (recomendado hoy):

  ```bash
  pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  ```
* (Si usas otra serie CUDA, consulta la nota de versiones en *releases* de JAX). ([GitHub][3])

4. **WSL2 + GPU (Windows)**

   * Instala el **driver NVIDIA para Windows** (incluye soporte para WSL); **no** instales un driver Linux dentro de WSL. ([NVIDIA Docs][4])

---

## 🧠 API del ambiente

```python
from env import MinesweeperJAX, build_jit_env
import jax, jax.numpy as jnp

key = jax.random.PRNGKey(0)
env = MinesweeperJAX(H=16, W=16, mine_prob=0.15625, context_radius=1)

reset_jit, observe_jit, mask_jit, step_jit = build_jit_env(env)

B = 2048
state = reset_jit(key, B)               # B tableros en paralelo
obs   = observe_jit(state)               # (B, C, H, W)
mask  = mask_jit(state)                  # (B, H*W) acciones válidas

actions = jnp.argmax(mask, axis=1)       # política tonta
state, reward, done = step_jit(state, actions)
```

### Detalles de diseño

* **Cálculo de vecinos**: se usa `lax.reduce_window` para sumar vecinos (8-conectado), evitando *integer convolutions* sobre cuDNN que no están soportadas tal cual (INT8/s8/s32 restricciones). ([Stack Overflow][5])
* **Pytree**: `MinesState` se registra con `jax.tree_util.register_pytree_node` para poder pasar/retornar el estado por `jit/scan`. ([JAX Docs][6])
* **`jit` y estáticos**: el wrapper `reset_jit` marca el **batch size** `B` como `static_argnames=('B',)` para especializar la compilación por forma. ([JAX Docs][7])

---

## 👀 Observación (formato)

* 9 canales *one-hot* (0..8) **solo** donde la celda está revelada.
* 1 canal de **no reveladas**.
* 1 canal de **banderas**.
* **Contexto** opcional (radio `R`):

  * canales “*shifted*” con el número visible de cada vecino dentro de la ventana (si no está revelado, 0);
  * 2 canales de resumen: conteo de **no reveladas** y de **banderas** en la ventana (padding SAME por `reduce_window`/conv).

---

## 🎬 Grabar un episodio a MP4

```bash
python record.py \
  --H 16 --W 16 --R 1 --T 300 \
  --policy random \
  --out videos/mines_episode.mp4
```

* Usa **Pillow** para dibujar el tablero y **imageio-ffmpeg** para escribir el video (incluye binarios ffmpeg cross-plataforma).
  Si alguna guía antigua usa `ImageDraw.textsize`, ten en cuenta que esa API fue deprecada/retirada; en este repo se usa `textbbox`. ([Pillow (PIL Fork)][8])

---

## 🕹️ UI interactiva (Streamlit)

```bash
streamlit run app_streamlit.py
```

* Click sobre celdas, revela/flaggea, y a la derecha verás la **observación** (canales) del agente en tiempo real.
* La app usa el componente **`streamlit-image-coordinates`** para leer coordenadas de clicks en imágenes. ([PyPI][9])

---

## 🧪 Benchmarks de estrés

Ejemplo (GPU):

```bash
python stress_bench.py --H 16 --W 16 --R 1 --T 128 \
  --mode scan --policy first_valid --batches 256,512,1024,2048,4096
```

El modo `scan` hace *unroll* por `lax.scan` para secuencias de pasos sin salir de JAX.
En una RTX 4070 Laptop GPU deberías ver **millones de steps/s**. Este orden de magnitud es coherente con motores JAX paralelos (p.ej., Brax reporta *millions of steps per second* por acelerador en sus envs). ([GitHub][10])

---

## ⚙️ Políticas incluidas (demo)

* `first_valid`: elige la primera acción válida
* `random`: acción válida al azar

Puedes implementar la tuya y *jit*-compilarla. Mantén todo en JAX para no romper la *staging*.

---

## 🧩 Entrenar un agente (pista)

Aún no hay *trainer* en el repo, pero el entorno está listo para integrarlo con tu framework JAX (Flax/Optax, etc.). Mantén la **política** batched `(B, ...) -> (B,)` y compila el `rollout` con `jit` para throughput máximo.

---

## 🛟 Troubleshooting (errores comunes)

* **`externally-managed-environment` al instalar**
  Crea/activa un `venv` y vuelve a instalar (ver PEP 668). ([TurnKey Linux][2])

* **JAX no usa GPU / cuDNN init failed / driver insuficiente**

  * Revisa que instalaste el **driver NVIDIA en Windows** (no en WSL) y `jax[cuda12]` en el venv de WSL. ([NVIDIA Docs][4])
  * Evita convs enteras: este repo usa `reduce_window` para vecinos; si introduces `lax.conv_general_dilated` con enteros podrías topar el error **“Can't lower integer convolutions to CuDNN”**; convierte a float o usa `reduce_window`. ([GitHub][11])

* **Lento en WSL**
  Asegúrate de **no** ejecutar desde `/mnt/c/...`. Trabaja bajo `\\wsl$\Distro\home\...` para mejor I/O. ([Microsoft Learn][1])

* **`jit` con argumentos no-array**
  Si un `jit` recibe un objeto Python (no array) que cambia el control de flujo, márcalo como **estático** con `static_argnames`/`static_argnums` o reestructura la función. ([JAX Docs][7])

---

## 📊 Métricas que puedes reportar

* `steps/s` = `B * pasos_por_segundo`
* `cells/s` ≈ `steps/s * (H*W)` (aprox. de trabajo por tablero)
* % de episodios terminados, recompensa media, tiempo total

---

## 🧾 Licencias / Créditos

* JAX & docs (Google/JAX) ([JAX Docs][7])
* *WSL GPU* (NVIDIA, Microsoft Docs) ([NVIDIA Docs][4])
* *WSL filesystems best practices* (MS Docs) ([Microsoft Learn][1])
* Pillow deprecations / imageio-ffmpeg ([Pillow (PIL Fork)][8])
* Streamlit components / docs ([PyPI][9])
* Brax (contexto de rendimiento en JAX) ([GitHub][10])

---

## 🧭 Roadmap (ideas)

* Integración con **Flax** y **Optax** (DQN/IMPALA/SAC-discreto).
* *Curriculum* de dificultad (probabilidad de mina adaptativa).
* Render **100% JAX** (frames como arrays) para *end-to-end GPU*.
* Tests de consistencia y *golden seeds*.

---

### Comandos rápidos

```bash
# 1) Activar entorno
source .venv/bin/activate

# 2) Demo rápida
python -c "from env import *; import jax; env=MinesweeperJAX(); r,o,m,s=build_jit_env(env); st=r(jax.random.PRNGKey(0), 1024); print(o(st).shape)"

# 3) Bench
python stress_bench.py --H 16 --W 16 --R 1 --T 128 --mode scan --policy first_valid --batches 256,512,1024,2048,4096

# 4) Video
python record.py --H 16 --W 16 --R 1 --T 300 --policy random --out videos/mines_episode.mp4

# 5) UI
streamlit run app_streamlit.py
```

¡Listo! Con esto tienes el ambiente, herramientas de visualización y *bench* para exprimir tu GPU al máximo y entender qué está “viendo” tu agente en cada paso.

[1]: https://learn.microsoft.com/en-us/windows/wsl/filesystems?utm_source=chatgpt.com "Working across Windows and Linux file systems"
[2]: https://www.turnkeylinux.org/blog/python-externally-managed-environment?utm_source=chatgpt.com "Python PEP 668 - working with \"externally managed ..."
[3]: https://github.com/jax-ml/jax/releases?utm_source=chatgpt.com "Releases · jax-ml/jax"
[4]: https://docs.nvidia.com/cuda/archive/11.3.1/wsl-user-guide/index.html?utm_source=chatgpt.com "CUDA on WSL User Guide"
[5]: https://stackoverflow.com/questions/75608323/how-do-i-solve-error-externally-managed-environment-every-time-i-use-pip-3?utm_source=chatgpt.com "externally-managed-environment\" every time I use pip 3? ..."
[6]: https://docs.jax.dev/en/latest/working-with-pytrees.html?utm_source=chatgpt.com "Working with pytrees"
[7]: https://docs.jax.dev/en/latest/jit-compilation.html?utm_source=chatgpt.com "Just-in-time compilation"
[8]: https://pillow.readthedocs.io/en/stable/deprecations.html?utm_source=chatgpt.com "Deprecations and removals - Pillow (PIL Fork) - Read the Docs"
[9]: https://pypi.org/project/streamlit-image-coordinates/?utm_source=chatgpt.com "streamlit-image-coordinates"
[10]: https://github.com/google/brax?utm_source=chatgpt.com "google/brax: Massively parallel rigidbody physics ..."
[11]: https://github.com/google/jax/issues/10128?utm_source=chatgpt.com "jax.numpy.convolve outputs floats for integer inputs #10128"

---

## Nombre del paquete e import

- El nombre publicado en PyPI es `minesweeper-env-rl`, por lo que debes instalarlo con `pip install minesweeper-env-rl`.
- Para importarlo en Python usá guiones bajos: `from minesweeper_env_rl import MinesweeperJAX`.
- Si tenés código heredado que todavía usa `from RL_enviroment import ...`, considerá actualizarlo a `minesweeper_env_rl` para evitar confusiones.

---

## Automatización de releases a PyPI

- El workflow `.github/workflows/publish.yml` publica `minesweeper-env-rl` en PyPI cada vez que hay push a `main` (o cuando se ejecuta manualmente vía `workflow_dispatch`). Antes de construir los artefactos corre `python minesweeper_env_rl/stress_bench.py --H 8 --W 8 --T 16 --mode scan --policy first_valid --batches 64,128` para comparar el rendimiento entre versiones.
- El script `scripts/bump_version.py` actualiza `minesweeper_env_rl/pyproject.toml` y `minesweeper_env_rl/__init__.py`. Por defecto incrementa el `minor`, pero si el último mensaje de commit incluye `[major]`, `#major`, `BREAKING CHANGE` o `!:` sube el `major`; si incluye `[patch]`, `[fix]`, `#patch` o `#fix` sube únicamente el `patch`.
- La acción `stefanzweifel/git-auto-commit-action` commitea el nuevo número de versión (`chore: release ... [skip ci]`) para que cada build quede registrado.
- Para publicar necesitas un secreto de repo llamado `PYPI_API_TOKEN` con un *project token* de PyPI que tenga permisos de upload sobre `minesweeper-env-rl`.
