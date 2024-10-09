
---
---
# iSpeech: Improving inner speech decoding by hybridisation of bimodal EEG and fMRI data
---
---

This repository supports the publicly-available dataset produced by Liwicki _et al._ [1], and contains the preprocessed data and Python code to support the publication of Wellington _et al._ [2].

EEG-fMRI bimodal model classification performance is investigated for four participants' data, performing inner speech of eight selected words, comprising four words each from two categories, 'numbers' and 'social':

`four` `three` `ten` `six`

`daughter` `father` `wife` `child`

Further paradigm details can be found in the associated publication [2].

The script `EMBC_fusion_classification_early_late.py` will run the complete classification pipeline, for both 'early fusion' and 'late fusion' methodologies. The script is designed to run out-of-the-box, but some programming knowledge may be required to change parameter configurations, e.g.

The participant number, and option to plot the projected data struction, are variables that can be set under `Configurable parameters`. Code is provded as-is, and researchers are invited to experiment with models, parameters, and different data preprocessing methdologies (see [1]).

The directory `iSpeech_proc` contains the preprocessed fMRI and EEG data of the participants, with the following directory structure:


```
iSpeech_proc
  ├── EEG-proc
  │     └── epoched
  │             ├── subject01_eeg-epo.fif.gz
  │             ├── ...
  │             └── subject05_eeg-epo.fif.gz
  └── fMRI-proc
        ├── labels
        │       ├── single_condition_subject1_session1.txt
        │       ├── ...
        │       └── single_condition_subject5_session2.txt
        │       
        ├── sub-01
        │       ├── single_trial_sess1
        │       │       ├── beta_0001.nii
        │       │       ├── ...
        │       │       └── sub2_sess1_betas_4D.nii
        │       └── single_trial_sess2
        │               ├── beta_0001.nii
        │               ├── ...
        │               └── sub2_sess2_betas_4D.nii
        ├── ...
        │
        └── sub-05
                └── ...
```

We hope for the artifacts resulting from iSpeech to be of some utility for future researchers, due to the sparsity of similar open-access research. As such, this code and (preprocessed) data is made freely available for all academic and research purposes (non-profit).

---

#### REFERENCING

---

If you use the code or data, please reference:

* S. Wellington, H. Wilson, F. Simistira Liwick, V. Gupta, R. Saini _et al._ "Improving inner speech decoding by hybridisation of bimodal EEG and fMRI data." Proceedings of the 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC) (2024) [in press].

---

#### LEGAL

---

For the purpose of open access, the author has applied a Creative Commons Attribution (CC BY) license to any Author Accepted Manuscript version arising: https://creativecommons.org/licenses/by/4.0/

---

#### ACKNOWLEDGEMENTS

---

This research was funded by:

--- The United Kingdom Research Institute (UKRI; grant EP/S023437/1)
--- The Engineering and Physical Sciences Research Council (EPSRC; grant EP/S515279/1)
--- The Grants for Excellent Research Projects Proposals of SRT.ai 2022.

---
---

[1] F. Simistira Liwicki, V. Gupta, R. Saini, K. De, N. Abid, S. Rakesh, S. Wellington, H. Wilson, M. Liwicki, and J. Eriksson, “Bimodal electroencephalography-functional magnetic resonance imaging dataset for inner-speech recognition,” Scientific Data, vol. 10, no. 1, p. 378, 2023.

[2] S. Wellington, H. Wilson, F. Simistira Liwick, V. Gupta, R. Saini, K. De, N. Abid, S. Rakesh, J. Eriksson, O. Watts, X. Chen, D. Coyle, M. Golbabaee, M. J. Proulx, M.  Liwicki, E. O'Neill, B. Metcalfe, "Improving inner speech decoding by hybridisation of bimodal EEG and fMRI data." Proceedings of the 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (2024) [in press].