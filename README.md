# 🩺 MediAI — Plateforme d'IA agentique pour le diagnostic du cancer du sein

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

**MediAI** est une plateforme intelligente d'aide à la décision pour le diagnostic du cancer du sein, basée sur une architecture multi-agents modulaire. Elle combine des modèles de vision fine-tunés, une fusion multimodale, un mécanisme de RAG médical et un grand modèle de langage pour générer des rapports cliniques structurés et interprétables à partir de mammographies.

---

## 📋 Table des matières

- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Technologies](#-technologies)
- [Résultats](#-résultats)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Auteur](#-auteur)
- [Licence](#-licence)

---

## ✨ Fonctionnalités

- 🔍 **Détection automatique des lésions** via YOLO11m fine-tuné sur le dataset DMID
- 🏷️ **Classification en 3 classes** (Normal, Bénin, Malin) avec Swin Transformer
- 📐 **Extraction des métriques morphologiques** (taille, aire, périmètre, circularité)
- 🧬 **Fusion multimodale** (images + données cliniques) via BioMedCLIP
- 📚 **Recherche contextuelle (RAG)** dans Qdrant avec BGE-M3 et guidelines NCCN/AGO
- 🧠 **Raisonnement clinique structuré** généré par Qwen2.5-7B via Ollama
- 📊 **Attribution automatique du BI-RADS** basée sur le score de malignité
- 🖼️ **Génération d'images annotées** (ellipses rouges pour Malin, vertes pour Bénin)
- 📄 **Rapports cliniques structurés** exportables en PDF
- 💬 **Assistant conversationnel** pour interroger le rapport
- ⚡ **Suivi en temps réel** via WebSocket
- 🎨 **Interface web React** avec tableau de bord, historique et statistiques

---

## 🏗️ Architecture

MediAI repose sur une architecture multi-agents orchestrée par **LangGraph**. Le pipeline complet se déroule en plusieurs étapes, de l'image brute à la génération du rapport clinique.
┌─────────────────┐
│ Image PNG │
│ (Mammographie) │
└────────┬────────┘
│
▼
┌─────────────────────────────────────────────┐
│ Orchestrateur (LangGraph) │
└────────┬────────────────────────────────────┘
│
├──► Agent Vision (YOLO + Swin + Masquage)
│
├──► Agent Fusion (BioMedCLIP)
│
├──► Agent RAG (Qdrant + BGE-M3)
│
├──► Agent Raisonnement (Qwen2.5-7B)
│
└──► Agent Rapport (Structuration + Images annotées)
│
▼
┌───────────────────────┐
│ Rapport clinique │
│ (Markdown + PDF) │
└───────────────────────┘


### Couches de l'architecture

- **Frontend** : React 18 + TypeScript + Tailwind CSS
- **API Gateway** : FastAPI + Uvicorn (REST + WebSocket)
- **Orchestrateur** : LangGraph (graphe d'état partagé)
- **Agents IA** : Vision, Fusion, RAG, Raisonnement, Rapport
- **Services** : PostgreSQL (persistance), Qdrant (vectoriel), Ollama (LLM)

---

## 🛠️ Technologies

| Catégorie | Technologies |
| :--- | :--- |
| **Langages** | Python 3.12, TypeScript |
| **Backend** | FastAPI, Uvicorn, LangGraph, SQLAlchemy |
| **Vision** | PyTorch, YOLO11m, Swin Transformer, OpenCV |
| **Fusion multimodale** | BioMedCLIP |
| **LLM** | Qwen2.5-7B-Instruct (via Ollama) |
| **Embeddings** | BGE-M3 (Sentence-Transformers) |
| **RAG** | Qdrant |
| **Base de données** | PostgreSQL |
| **Frontend** | React 18, Tailwind CSS, Zustand, Recharts |
| **Conteneurisation** | Docker, Docker Compose |

---

## 📊 Résultats

### Performances des modèles de vision

| Modèle | Métrique | Valeur |
| :--- | :--- | :--- |
| **YOLO11m** (détection) | mAP50 | 87,9 % |
| | Rappel (lésions malignes) | 91,1 % |
| | Temps d'inférence | 9,3 ms/image |
| **Swin-Base** (classification) | Précision globale | **98,43 %** |
| | Précision (Normal) | 100 % |
| | Précision (Bénin / Malin) | 96,7 % / 96,3 % |

### Performances du pipeline

| Étape | Temps moyen |
| :--- | :--- |
| Prétraitement + Détection + Classification | 2,5 s |
| Fusion multimodale + RAG | 2,2 s |
| Raisonnement clinique (Qwen2.5) | 25 à 45 s |
| Génération du rapport | 1,0 s |
| **Total** | **30 à 50 s** |

### Évaluation qualitative du raisonnement clinique

| Critère | Score moyen |
| :--- | :--- |
| Cohérence clinique | 9,2 / 10 |
| Exhaustivité | 9,5 / 10 |
| Clarté et lisibilité | 9,0 / 10 |
| Absence d'hallucinations | 8,5 / 10 |
| Pertinence des recommandations | 9,3 / 10 |
| **Moyenne** | **9,1 / 10** |

