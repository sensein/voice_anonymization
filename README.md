# Voice Cloning for Child Speaker Anonymization

This project explores the feasibility of using voice cloning as a solution for anonymizing the voices of child speakers. The aim is to protect the privacy of minors by altering their voices while retaining speech utility (including intelligibility and naturalness).

## 🛠️ Setup

Follow these steps to set up the environment:

1. **Create a new conda environment:**
   ```bash
   conda create -n child_speaker_anonymization python=3.10
   ```

2. **Activate the environment:**
   ```bash
   conda activate child_speaker_anonymization
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

If you use OpenMind, the data for our first experiments is stored here: ```<TODO: ADD>```. Soon(ish) we will set up a datalad repo for taking care of data versioning (**TODO**).

## 🚀 Quick Start

To run the project, simply execute:

```bash
python main.py
```

## 📁 Project Files

Access the project files and related data [here](https://drive.google.com/drive/folders/1vJYu2FN2aKeHd_fB6guxvQWoyspLdqV3?usp=sharing).


### TODO:
-[] SETUP DATALAD WITH GOOGLE DRIVE (SUBDATASETS) - for now you can find it at /nese/mit/group/sig/projects/fabiocat/children_voice_anonymization

-[] VOICE CLONING PIPELINE

-[] ASR BENCHMARKING

-[] SER BENCHMARKING

