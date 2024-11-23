# Gender-experiments
[![gender-experiments-world](https://github.com/PedestrianDynamics/gender-experiments/actions/workflows/code-quality.yml/badge.svg)](https://github.com/PedestrianDynamics/gender-experiments/actions/workflows/code-quality.yml)
[![Open Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gender-experiments.streamlit.app/)
[![DOI: Data](https://img.shields.io/badge/DOI-10.34735/ped.2024.1-B31B1B.svg)](https://doi.org/10.34735/ped.2024.1)
[![DOI: Code](https://zenodo.org/badge/DOI/10.5281/zenodo.12675716.svg)](https://doi.org/10.5281/zenodo.12675716)
[![DOI: Article](https://img.shields.io/badge/DOI-10.34735/ped.2024.1-blue.svg)](https://doi.org/10.34735/ped.2024.1)
    

## Data
Single-file movement experiments conducted under comparable conditions across five distinct countries:
- 🇦🇺 Australia
- 🇨🇳 China
- 🇯🇵 Japan
- 🇩🇪 Germany 
- 🇵🇸 Palestine 

The primary variable altered in these experiments are density (number of people increased gradually) and the gender composition, which includes categories such as male-only, female-only, randomly mixed, and alternatively mixed runs. 

A more detailed description of the data can be found [here](https://doi.org/10.34735/ped.2024.1).

Preprint can be found [here](http://dx.doi.org/10.2139/ssrn.4900271).


##  RUNNING 

You can run the hosted version of this app by clicking the streamlit badge above.
Alternatively, you can also run it locally:

1. Install requirements

```bash
pip installl -r requirements
```

2. Run the app

```
streamlit run app.py
```

## ACKNOWLEDGMENT

We thank all the colleagues who helped with the organization of the experiments and extraction of the trajectories, as well as the curation of the data. Special thanks go to Alica Kandler for the curation and data quality of the German data, Reza Shahbad for managing the logistics of the experiments conducted in Australia, and Maziar Yazdani for the curation of the Australian data and Shi Dongdong (experiments in China).

Claudio Feliciani and Xiaolu Jia express their gratitude for the funding received through the JST-Mirai Program grant numbers JPMJMI20D1 and the JSPS KAKENHI grant numbers JP20K14992, JP23K13521, and JP21K14377. They also appreciate the support from www.jikken-baito.com and the Meguro Senior Employment Center in recruiting participants.

Milad Haghani acknowledges the support from the Australian Research Council, with grant number DE210100440.

Jian Ma acknowledges funding provided by the National Natural Science Foundation of China (Nos. 72104205) and the National Key Research and Development Program of China (No. 2022YFC3005205).

The German experiment, detailed at https://doi.org/10.34735/ped.2021.2, was part of an experimental series for the CroMa and CrowdDNA projects. These experiments took place at the Mitsubishi Electric Halle in Düsseldorf, Germany, in 2021.
