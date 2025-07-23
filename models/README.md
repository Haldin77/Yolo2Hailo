# Guide : Utiliser un modèle HEF YOLOv8‑Seg avec DeGirum PySDK et Hailo‑8 sur Raspberry Pi 5

Ce guide s’appuie sur le tutoriel **“Using DeGirum PySDK, DeGirum Tools, and Hailo Hardware”** et montre comment :

1. Préparer votre environnement Raspberry Pi 5 + Hailo‑8 / Hailo‑8L.
2. Installer la boîte à outils DeGirum.
3. Organiser votre *zoo* local de modèles et déployer votre modèle compilé (`yolov8n_seg`).

---

## Table des matières

1. [Introduction](#introduction)
2. [Prérequis](#prérequis)
3. [Installation](#installation)
4. [Organisation du dossier `models/`](#organisation-du-dossier-models)
5. [Exécution et configuration des notebooks Jupyter](#exécution-et-configuration-des-notebooks-jupyter)
6. [Exemple de script d’inférence](#exemple-de-script-dinférence)
7. [Ressources supplémentaires](#ressources-supplémentaires)

---

## Introduction

DeGirum offre une suite d’outils permettant de simplifier le développement et le déploiement d’applications **Edge AI** :

* **DeGirum PySDK** : bibliothèque centrale pour intégrer l’inférence IA dans vos applications.
* **DeGirum Tools** : utilitaires pour le benchmark, le streaming et l’interaction avec le *model zoo* DeGirum.

Ces outils sont **agnostiques au matériel** ; vous pouvez donc créer des solutions flexibles et évolutives sur plusieurs plateformes, notamment **Hailo‑8** et **Hailo‑8L**.

---

## Prérequis

| Élément                           | Version / Détails                                            |
| --------------------------------- | ------------------------------------------------------------ |
| **Hailo Tools**                   | Installées et configurées (voir docs Hailo)                  |
| **HailoRT Multi‑Process Service** | Activé : `sudo systemctl enable --now hailort.service`       |
| **Hailo Runtime**                 | Versions supportées par PySDK : **4.19.0 / 4.20.0 / 4.21.0** |
| **Python**                        | ≥ 3.9 (`python3 --version`)                                  |
| **Raspberry Pi OS**               | 64‑bits, à jour                                              |

---

## Installation

> **Astuce :** les étapes ci‑dessous reprennent celles du dépôt `hailo_examples` de DeGirum.

### 1. Cloner le dépôt d’exemples

```bash
git clone https://github.com/DeGirum/hailo_examples.git
cd hailo_examples
```

### 2. Créer un environnement virtuel

```bash
python3 -m venv degirum_env
source degirum_env/bin/activate   # Linux / macOS
# ou, sous Windows :
# degirum_env\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Ajouter l’environnement virtuel à Jupyter (optionnel)

Si vous prévoyez d’utiliser des notebooks Jupyter :

```bash
python -m ipykernel install --user --name=degirum_env --display-name "Python (degirum_env)"
```

### 5. Vérifier l’installation

```bash
python test.py
```

Ce script :

1. Affiche les infos système.
2. Vérifie la détection du matériel Hailo.
3. Charge un modèle d’exemple et lance une inférence.

Si tout se passe bien, votre environnement est prêt ! ✅

---

## Organisation du dossier `models/`

Pour que l’appel :

```python
model = dg.load_model(
    model_name="yolov8n_seg",
    inference_host_address="@local",
    zoo_url="/home/robucar/hailo_examples/models",
    device_type=["HAILORT/HAILO8"],
)
```

fonctionne, la structure attendue est la suivante :

```
/home/robucar/hailo_examples/models/
└── yolov8n_seg/
    ├── yolov8n_seg.hef            # Graphe compilé pour Hailo
    ├── yolov8n_seg.json           # Fichier de configuration DeGirum
    ├── crack.json                 # Label map (1 classe « crack »)
```

### Rôle de chaque fichier

| Fichier                       | Description                                                             |
| ----------------------------- | ----------------------------------------------------------------------- |
| **yolov8n\_seg.hef**          | Network Binary compilé via Hailo Dataflow Compiler.                     |
| **yolov8n\_seg.json**         | Configuration DeGirum (pré‑/post‑process, device, chemin vers le .hef). |
| **crack.json**                | Dictionnaire des labels (clé numérique → nom de classe).                |


> **Important :** le nom du dossier **et** le préfixe des fichiers doivent correspondre à `model_name` passé à `dg.load_model`.

#### Exemple minimal de `yolov8n_seg.json`

**Voir plus bas pour une version complète automatique.**

```json
{
  "ConfigVersion": 11,
  "Checksum": "<sha256_du_hef>",
  "DEVICE": [{
    "DeviceType": "HAILO8",
    "RuntimeAgent": "HAILORT",
    "SupportedDeviceTypes": "HAILORT/HAILO8"
  }],
  "PRE_PROCESS": [{
    "InputType": "Image",
    "InputN": 1,
    "InputH": 640,
    "InputW": 640,
    "InputC": 3,
    "InputPadMethod": "letterbox",
    "InputResizeMethod": "bilinear",
    "InputQuantEn": true
  }],
  "MODEL_PARAMETERS": [{
    "ModelPath": "yolov8n_seg.hef"
  }],
  "POST_PROCESS": [{
    "OutputPostprocessType": "SegmentationYoloV8",
    "LabelsPath": "crack.json",
    "OutputNumClasses": 1,
    "OutputConfThreshold": 0.3,
    "SigmoidOnCLS": true
  }]
}
```

#### Étapes détaillées de création manuelle

1. **Checksum** : exécutez `sha256sum yolov8n_seg.hef` et copiez la valeur dans la clé `Checksum`.
2. **ConfigVersion** : utiliser `11` (schéma courant).
3. **DEVICE / PRE\_PROCESS / MODEL\_PARAMETERS / POST\_PROCESS** : adaptez si vous changez la taille d’entrée, le nombre de classes ou le nom du `.hef`.

#### Génération automatique par script

```python
import json, hashlib
hef = "yolov8n_seg.hef"
checksum = hashlib.sha256(open(hef, "rb").read()).hexdigest()
config = {
  "ConfigVersion": 11,
  "Checksum": checksum,
  "DEVICE": [{
      "DeviceType": "HAILO8",
      "RuntimeAgent": "HAILORT",
      "SupportedDeviceTypes": "HAILORT/HAILO8"
  }],
  "PRE_PROCESS": [{
      "InputType": "Image",
      "InputN": 1,
      "InputH": 640,
      "InputW": 640,
      "InputC": 3,
      "InputPadMethod": "letterbox",
      "InputResizeMethod": "bilinear",
      "InputQuantEn": true
  }],
  "MODEL_PARAMETERS": [{"ModelPath": "yolov8n_seg.hef"}],
  "POST_PROCESS": [{
      "OutputPostprocessType": "SegmentationYoloV8",
      "LabelsPath": "crack.json",
      "OutputNumClasses": 1,
      "OutputConfThreshold": 0.3,
      "SigmoidOnCLS": true
  }]
}
with open("yolov8n_seg.json", "w") as f:
    json.dump(config, f, indent=2)
print("yolov8n_seg.json créé ✔")
```

*Le script calcule automatiquement le SHA‑256 du `.hef` et écrit le JSON prêt à l’emploi.*

```json
{
  "ConfigVersion": 10,
  "DEVICE": [{
    "DeviceType": "HAILO8",
    "RuntimeAgent": "HAILORT",
    "SupportedDeviceTypes": "HAILORT/HAILO8"
  }],
  "PRE_PROCESS": [{
    "InputType": "Image",
    "InputN": 1,
    "InputH": 640,
    "InputW": 640,
    "InputC": 3,
    "InputPadMethod": "letterbox"
  }],
  "MODEL_PARAMETERS": [{
    "ModelPath": "yolov8n_seg.hef"
  }],
  "POST_PROCESS": [{
    "OutputPostprocessType": "SegmentationYoloV8",
    "LabelsPath": "crack.json",
    "OutputNumClasses": 1,
    "OutputConfThreshold": 0.3
  }]
}
```

Si vous changez le nombre de classes ou le seuil de confiance, modifiez `OutputNumClasses`, `LabelsPath` et `OutputConfThreshold` en conséquence.

---

## Exécution et configuration des notebooks Jupyter

1. Lancez Jupyter :

   ```bash
   jupyter lab  # ou jupyter notebook
   ```
2. Dans l’interface, choisissez le kernel **“Python (degirum\_env)”** pour vos notebooks.
3. Les notebooks du dépôt `hailo_examples/notebooks/` couvrent :

   * Benchmark de latence/débit.
   * Streaming vidéo.
   * Interfaçage avec le *model zoo* DeGirum.

---

## Exemple de script d’inférence

```python
import degirum as dg

model = dg.load_model(
    model_name="yolov8n_seg",
    inference_host_address="@local",
    zoo_url="/home/robucar/hailo_examples/models",
    device_type=["HAILORT/HAILO8"],
)

result = model("images/test.jpg")
print(result)
```

Ajouter un chronomètre :

```python
import time
start = time.time()
for _ in range(30):
    _ = model("images/test.jpg")
print("Avg latency (ms):", (time.time()-start)/30*1000)
```

---

## Ressources supplémentaires

* **Documentation DeGirum PySDK** : [https://docs.degirum.com/pysdk](https://docs.degirum.com/pysdk)
* **Projet `hailo_examples`** : [https://github.com/DeGirum/hailo\_examples](https://github.com/DeGirum/hailo_examples)
* **Documentation Hailo** : [https://docs.hailo.ai](https://docs.hailo.ai)

---

*Fin du guide.*
