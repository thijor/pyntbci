# PyntBCI

Python Noise-Tagging Brain-Computer interface (PyntBCI) is a Python library for the noise-tagging brain-computer 
interface (BCI) project developed at the Donders Institute for Brain, Cognition and Behaviour, Radboud University, 
Nijmegen, the Netherlands. PyntBCI contains various signal processing steps and machine learning algorithms for BCIs 
that make use of evoked responses of the electroencephalogram (EEG), specifically code-modulated responses such as the 
code-modulated visual evoked potential (c-VEP). For a constructive review of this field, see [7].  

## Installation

To install PyntBCI, use:

	pip install pyntbci

## Getting started

Various tutorials and example analysis pipelines are provided in the `tutorials` and `examples/` folder, which operate 
on the datasets as provided below. 

## References

[1]: Thielen, J., van den Broek, P., Farquhar, J., & Desain, P. (2015). Broad-Band visually evoked potentials: 
re(con)volution in brain-computer interfacing. PLOS ONE, 10(7), e0133797. 
DOI: [10.1371/journal.pone.0133797](https://doi.org/10.1371/journal.pone.0133797)

[2]: Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2017). Re(con)volution: accurate response prediction for 
broad-band evoked potentials-based brain computer interfaces. In Brain-Computer Interface Research (pp. 35-42). 
Springer, Cham. DOI: [10.1007/978-3-319-64373-1_4](https://doi.org/10.1007/978-3-319-64373-1_4)

[3]: Desain, P. W. M., Thielen, J., van den Broek, P. L. C., & Farquhar, J. D. R. (2019). U.S. Patent No. 10,314,508. 
Washington, DC: U.S. Patent and Trademark Office. 
Link: [here](https://patentimages.storage.googleapis.com/40/a3/bb/65db00c7de99ec/US10314508.pdf)

[4]: Ahmadi, S., Borhanazad, M., Tump, D., Farquhar, J., & Desain, P. (2019). Low channel count montages using sensor 
tying for VEP-based BCI. Journal of Neural Engineering, 16(6), 066038. 
DOI: [10.1088/1741-2552/ab4057](https://doi.org/10.1088/1741-2552/ab4057)

[5]: Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a 
code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5), 056007. 
DOI: [10.1088/1741-2552/abecef](https://doi.org/10.1088/1741-2552/abecef)

[6]: Verbaarschot, C., Tump, D., Lutu, A., Borhanazad, M., Thielen, J., van den Broek, P., ... & Desain, P. (2021). A 
visual brain-computer interface as communication aid for patients with amyotrophic lateral sclerosis. Clinical 
Neurophysiology, 132(10), 2404-2415. DOI: [10.1016/j.clinph.2021.07.012](https://doi.org/10.1016/j.clinph.2021.07.012)

[7]: Martínez-Cagigal, V., Thielen, J., Santamaría-Vázquez, E., Pérez-Velasco, S., Desain, P., & Hornero, R. (2021). 
Brain–computer interfaces based on code-modulated visual evoked potentials (c-VEP): a literature review. Journal of 
Neural Engineering. DOI: [10.1088/1741-2552/ac38cf](https://doi.org/10.1088/1741-2552/ac38cf)

[8]: Thielen, J. (2023). Effects of Stimulus Sequences on Brain-Computer Interfaces Using Code-Modulated Visual 
Evoked Potentials: An Offline Simulation. In International Work-Conference on Artificial Neural Networks (pp. 555-568). 
Cham: Springer Nature Switzerland. DOI: [10.1007/978-3-031-43078-7_45](https://doi.org/10.1007/978-3-031-43078-7_45)

## Datasets

On the Radboud Data Repository (RDR) (https://data.ru.nl/):
* Thielen et al. (2018) Broad-Band Visually Evoked Potentials: Re(con)volution in Brain-Computer Interfacing. 
DOI: [10.34973/1ecz-1232](https://doi.org/10.34973/1ecz-1232)
* Ahmadi et al. (2018) High density EEG measurement. DOI: [10.34973/psaf-mq72](https://doi.org/10.34973/psaf-mq72)
* Ahmadi et al. (2019) Sensor tying. DOI: [10.34973/ehq6-b836](https://doi.org/10.34973/ehq6-b836)
* Thielen et al. (2021) From full calibration to zero training for a code-modulated visual evoked potentials brain 
  computer interface. DOI: [10.34973/9txv-z787](https://doi.org/10.34973/9txv-z787)

On Mother of all BCI Benchmarks (MOABB) (https://moabb.neurotechx.com/docs/index.html):
* [c-VEP dataset from Thielen et al. (2021)](
https://moabb.neurotechx.com/docs/generated/moabb.datasets.Thielen2021.html#moabb.datasets.Thielen2021)

## Contact

* Jordy Thielen (jordy.thielen@donders.ru.nl)

## Licensing

PyntBCI is licensed by the BSD 3-Clause License:

Copyright (c) 2021, Jordy Thielen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
