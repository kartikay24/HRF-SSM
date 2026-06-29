# HRF-SSM (ICML 2026)

This  repository contains the official implementation for the paper [A Spiking Heterogeneous Harmonic Resonate-and-Fire State Space Model for Time Series](https://openreview.net/forum?id=dDGxzaRkxO) by [Agrawal et al. 2026](https://kartikay24.github.io/) at the [SustainAI Lab](https://ayonborthakur.pages.dev/).

Evaluation on the 17 diverse datasets spans architectures from 3 different repositories:
1. Discretization exploration of BHRF on SHD datasets
2. SH2RFSSM (or spikinglinoss) for Long-Sequence Classification and Regression tasks (UEA + PPG) along with Human Activity Recognition datasets
3. SpikHRFSSM for Long-range Forecasting

- To train for the RNN setup
We extended [https://github.com/AdaptiveAILab/brf-neurons](https://github.com/AdaptiveAILab/brf-neurons) codebase to include learnable heterogeneities and various discretizations. You may follow the env setup steps from this codebase and run it on the Spiking Heidelberg dataset (SHD).

```
python3 train.py --data shd
```

- To train for the SSM setup
You can clone [https://github.com//tk-rusch/linoss](https://github.com//tk-rusch/linoss) and set up the environment and datasets from this codebase. Further, you can run the following script for results on the EigenWorms dataset (default).

```
python3 run_experiment.py
```

For forecasting, you may replace the obtained HRFSSM layer with the entire FFT-IFFT Block in [https://github.com/WWJ-creator/SpikF](https://github.com/WWJ-creator/SpikF).

# Citation
```bibtex
@inproceedings{
agrawal2026,
title={A Spiking Heterogeneous Harmonic Resonate-and-Fire State Space Model for Time Series},
author={Kartikay Agrawal and Vaishnavi Nagabhushana and Abhijeet Vikram and Vedant Sharma and Ayon Borthakur},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=dDGxzaRkxO}
}
}
@inproceedings{
agrawal2025a,
title={A Second-Order Spiking{SSM} for Wearables},
author={Kartikay Agrawal and Abhijeet Vikram and Vedant Sharma and Vaishnavi N and Ayon Borthakur},
booktitle={NeurIPS 2025 Workshop on Learning from Time Series for Health},
year={2025},
url={https://openreview.net/forum?id=hv52KEOshb}
}
}
```

