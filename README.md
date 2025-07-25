# YOLOv8‑Seg best.pt → HEF Hailo‑8

Projet réalisé au sein du laboratoire CRIStAL — Université de Lille.

**Déploiement sur Raspberry Pi 5 + Hailo‑8 / Hailo‑8L**
*Version : Juillet 2025*

---

## Table des matières

1. [Prérequis Système](#prérequis-système)
2. [Installation de Docker](#installation-de-docker)
3. [Exécution du conteneur Hailo AI Suite](#exécution-du-conteneur-hailo-ai-suite)
4. [Export du modèle YOLOv8‑Seg en ONNX (hors Docker)](#export-du-modèle-yolov8‑seg-en-onnx-hors-docker)
5. [Compilation du modèle en HEF (dans le conteneur)](#compilation-du-modèle-en-hef-dans-le-conteneur)

---

## Prérequis Système

* **OS** : Ubuntu 20.04 / 22.04 (64 bits)
* **RAM** : ≥ 16 Go (32 Go recommandés)
* **Docker** : `docker.io` 20.10.07 ou `docker-ce` 20.10.6

---

## Installation de Docker

### 1. Ajout de l’utilisateur au groupe `docker`

```bash
sudo usermod -aG docker ${USER}
# puis déconnectez-vous et reconnectez-vous
```

### 2. (Optionnel) Support NVIDIA pour émulation/quantification

Configure the production repository:

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Optionally, configure the repository to use experimental packages:

```
sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Update the packages list from the repository:

```
sudo apt-get update
```

Install the NVIDIA Container Toolkit packages:

```
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

sudo systemctl restart docker
```

* GPU compatible Pascal/Turing/Ampere (Titan X Pascal, GTX 1080 Ti, RTX 2080 Ti, RTX A4000…)
* Driver NVIDIA 525

---

## Exécution du conteneur Hailo AI Suite

1. **Téléchargez** la suite logicielle et le driver PCIe depuis le Developer Zone.
2. **Installez** le driver PCIe :

   ```bash
   sudo dpkg -i <pcie_driver>.deb
   sudo reboot
   ```
3. **Dézippez** et lancez le conteneur :

   ```bash
   unzip hailo_ai_sw_suite_<version>.zip
   ./hailo_ai_sw_suite_docker_run.sh
   ```
4. **Vérifiez** les paquets Hailo et l’aide CLI :

   ```bash
   pip list | grep hailo
   hailo -h
   ```
5. **Options** du script :

   * `--help` pour afficher l’aide
   * `--hailort-enable-service` pour activer le service HailoRT multi-process
   * `--resume` pour reprendre un conteneur existant
   * `--override` pour supprimer et recréer le conteneur

---

## Export du modèle YOLOv8‑Seg en ONNX (hors Docker)

1. Créez et activez un environnement virtuel :

   ```bash
   python -m venv venv && source venv/bin/activate
   ```
2. Installez les dépendances :

   ```bash
   pip install "ultralytics>=8.2" onnx onnxsim
   ```
3. Exportez le modèle :

   ```bash
   yolo task=segment mode=export \
        model=best.pt \
        format=onnx \
        imgsz=640 \
        opset=12
   mv best.onnx bestcrack.onnx
   ```

---

## Compilation du modèle en HEF (dans le conteneur)

### Structure du projet

Placez dans un dossier (ex. `hailo_project/`):

```
hailo_project/
├── bestcrack.onnx
└── calib_imgs/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

### Compilation en ligne de commande

```bash
cd hailo_project
# depuis l’hôte :
./hailo_ai_sw_suite_docker_run.sh --resume
# ou, si déjà dans le conteneur :
hailo_docker

hailomz compile yolov8m_seg \
  --ckpt=bestcrack.onnx \
  --hw-arch=hailo8 \
  --calib-path=calib_imgs \
  --classes=1 \
  --performance
```

> **Note** : remplacez `--hw-arch=hailo8` par `hailo8l` pour Hailo‑8L et adaptez `--classes` selon votre jeu de données.

### Résultats attendus

* `yolov8m_seg.hef`
* `yolov8m_seg.runtime.json`
* `compile_report.html`

## Guide d'utilisation sur Raspberry Pi 5

Pour les instructions détaillées de déploiement sur Raspberry Pi 5, consultez le fichier [Guide](https://github.com/Haldin77/Yolo2Hailo/edit/main/models/README.md).

