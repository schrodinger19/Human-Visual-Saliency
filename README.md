# Human Visual Saliency Prediction
### Course Project — CGS401 (Cognitive Science, IIT Kanpur)
**Duration:** Aug’24 – Oct’24  
**Instructor:** Prof. Devpriya Kumar  

---

## Objective

Human visual saliency refers to the regions in an image that naturally attract human attention — determined through eye-tracking experiments and cognitive visual mechanisms.

The goal of this project is to model human visual attention as a **pixel-wise regression task**, predicting *saliency maps* that approximate where a human observer would look in a scene.

We implement a **Convolutional Neural Network (CNN)** that reconstructs these saliency maps from raw images, connecting computational vision with cognitive science.

---

## Motivation

Humans do not process every pixel equally. Our visual system performs **selective attention**, focusing on salient regions such as faces, motion, or high contrast.  
Modeling this behavior computationally has applications in:

- Image and video compression
- Autonomous driving
- Human-computer interaction and eye-tracking
- Robotics and visual question answering

---

## Dataset — SALICON (Saliency in Context)

**Total images:** ~10,000 (train) + 5,000 (validation/test)  
**Resolution:** Approximately 480×640  
**Format:** RGB images paired with human attention maps

| Component | Description |
|------------|-------------|
| **Images** | Sourced from the Microsoft COCO dataset; natural scenes with people, animals, and objects. |
| **Saliency Maps** | Continuous-valued attention maps estimated from human mouse-contingent annotation. |
| **Fixation Maps** | Binary maps indicating fixation points (used for evaluation metrics). |

**Download links:**
- Official SALICON dataset: [https://salicon.net/download/](https://salicon.net/download/)
- SALICON-mini (Kaggle, for quick testing): [https://www.kaggle.com/datasets/sshikamaru/salicon-mini](https://www.kaggle.com/datasets/sshikamaru/salicon-mini)


