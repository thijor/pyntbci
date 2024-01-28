# PyntBCI

The Python Noise-Tagging Brain-Computer interface (PyntBCI) library is a Python toolbox for the noise-tagging brain-computer interfacing (BCI) project developed at the Donders Institute for Brain, Cognition and Behaviour, Radboud University, Nijmegen, the Netherlands. PyntBCI contains various signal processing steps and machine learning algorithms for BCIs that make use of evoked responses of the electroencephalogram (EEG), specifically code-modulated responses such as the code-modulated visual evoked potential (c-VEP). For a constructive review of this field, see:

* Martínez-Cagigal, V., Thielen, J., Santamaría-Vázquez, E., Pérez-Velasco, S., Desain, P., & Hornero, R. (2021). Brain–computer interfaces based on code-modulated visual evoked potentials (c-VEP): a literature review. Journal of Neural Engineering. DOI: [10.1088/1741-2552/ac38cf](https://doi.org/10.1088/1741-2552/ac38cf)

## Installation

To install PyntBCI, use:

	pip install pyntbci

## Getting started

Various tutorials and example analysis pipelines are provided in the `tutorials/` and `examples/` folder, which operate on limited preprocessed data as provided with PyntBCI. Furthermore, please find various pipelines for several open-access datasets below in the `pipelines/` folder.

## Referencing

When using PyntBCI, please reference at least one of the following:

* Thielen, J., van den Broek, P., Farquhar, J., & Desain, P. (2015). Broad-Band visually evoked potentials: re(con)volution in brain-computer interfacing. PLOS ONE, 10(7), e0133797. DOI: [10.1371/journal.pone.0133797](https://doi.org/10.1371/journal.pone.0133797)
* Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5), 056007. DOI: [10.1088/1741-2552/abecef](https://doi.org/10.1088/1741-2552/abecef)

## Contact

* Jordy Thielen (jordy.thielen@donders.ru.nl)

## Licensing

PyntBCI is licensed by the BSD 3-Clause License:

Copyright (c) 2021, Jordy Thielen All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
