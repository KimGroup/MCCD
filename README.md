# 🧠 Multi-Core Circuit Decoder (MCCD)

This repository accompanies the paper:  
**"Learning to decode logical circuits"**  
by Yiqing Zhou, Chao Wan, Yichen Xu, Jin Peng Zhou, Kilian Q. Weinberger, and Eun-Ah Kim.  
📄 arXiv: [insert link here]

We introduce **MCCD**, a modular machine learning-based decoder designed for logical quantum circuits. MCCD handles both single- and entangling logical gates using gate-specific decoder modules. 

---
# Install MCCD
```bash
pip install mccd
```

# Curriculum Training

## Train single-qubit modules
```bash
bash train_1q.sh
```

## Train two-qubit modules
```bash
bash train_2q.sh
```
