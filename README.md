## Combinatorial Diffusion Auction for Crowdsourcing Internet of Things

---

### About

---

Code for No-Diffusion-VCG, Diffusion-VCG, Diff-CRA-HM and Diff-CRA-HT mechanisms for procurement auction via social networks (mechanisms mentioned in the the paper 'Combinatorial Diffusion Auction for Crowdsourcing Internet of Things'). 

Our new proposed Diff-CRA-HM and Diff-CRA-HT comprehensively outperform the classic mechanisms (VCG and its extensions, mechanisms held in local markets). 

In this paper, we firstly introduce the popluar diffusion auction model into the crowdsouring IoT resources scenarios. At the same time, we detailed analyzed the disadvantages of the previous works ['Selling multiple items via social networks'](https://dl.acm.org/doi/10.5555/3237383.3237400) and ['Strategy-proof and non-wasteful multi-unit auction via social network'](https://ojs.aaai.org/index.php/AAAI/article/view/5579/5435). Based on the framework for monotonic allocation rule with critical bid payment, we give an instance mechanism for the crowdsourcing procurement auctions on social networks.

---

### Files

```
├─README.md
├─requirements.txt
├─src
|  ├─attributes.py
|  ├─Core_Crowd_HM.py
|  ├─Core_Crowd_HT.py
|  ├─Crowd_Networks_Simulation.py
|  ├─Crowd_Size_Simulation.py
|  ├─Crowd_Task_Scale_Simulation.py
|  ├─Network_Load.py
|  ├─plot_figs.py
|  ├─Realistic_Networks_Heterogeneous_Simulation.py
|  ├─Realistic_Networks_Homogenous_Simulation.py
|  ├─__init__.py
|  ├─__pycache__
|  |      └attributes.cpython-36.pyc
├─datasets
|    ├─amazon.txt
|    ├─facebook_dataset1.txt
|    ├─facebook_dataset2.txt
|    ├─facebook_dataset3.txt
|    ├─github_network.txt
|    └twitter_dataset.txt
```

---

Diff-CRA-HM and Diff-CRA-HT mechanisms are implemented in Python. All the files are as follows:



---

### Note

- Firstly install all required modules for this project: `pip install -r requirements.txt`.
- `datasets` folder includes six real social network datasets from SNAP and Network Repository.
- run `python Realistic_Networks_Homogeneous_Simulation.py` and `python Realistic_Networks_Heterogeneous_Simulation.py` for simulation results of Table 2 and 3 in our paper. 
- run `python Crowd_Networks_Simulation.py` can get the Figure 4 and 5.
- run `python Crowd_Task_Size_Simulation.py` can get the Figure 6 and 7.
- run `python Crowd_Size_Simulation.py` can get the Figure 8 and 9. 
- `Core_Crowd_HM.py` gives the implementation of mechanisms: ND-VCG, D-VCG and Diff-CRA-HM while  `Core_Crowd_HT.py` shows the Diff-CRA-HT mechanism and the greedy algorithm in local market. 

