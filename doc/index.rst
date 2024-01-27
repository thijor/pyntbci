PyntBCI
=======

The Python Noise-Tagging Brain-Computer interface (PyntBCI) library is a Python toolbox for the noise-tagging brain-computer interfacing (BCI) project developed at the Donders Institute for Brain, Cognition and Behaviour, Radboud University, Nijmegen, the Netherlands. PyntBCI contains various signal processing steps and machine learning algorithms for BCIs that make use of evoked responses of the electroencephalogram (EEG), specifically code-modulated responses such as the code-modulated visual evoked potential (c-VEP). For a constructive review of this field, see [mar2021]_.

When using PyntBCI, please reference at least one of the following articles: [thi2015]_, [thi2017]_, [thi2021]_.

Installation
------------

To install PyntBCI, use:

	pip install pyntbci

Getting started
---------------

Various tutorials and example analysis pipelines are provided in the `tutorials/` (under Getting Started) and `examples/` (under Examples) folder, which operate on limited preprocessed data as provided with PyntBCI. Furthermore, please find various pipelines for several of the datasets referenced below in the `pipelines/` folder.

References
----------

.. [thi2015] Thielen, J., van den Broek, P., Farquhar, J., & Desain, P. (2015). Broad-Band visually evoked potentials: re(con)volution in brain-computer interfacing. PLOS ONE, 10(7), e0133797. DOI: `10.1371/journal.pone.0133797 <https://doi.org/10.1371/journal.pone.0133797>`_

.. [thi2017] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2017). Re(con)volution: accurate response prediction for broad-band evoked potentials-based brain computer interfaces. In Brain-Computer Interface Research (pp. 35-42). Springer, Cham. DOI: `10.1007/978-3-319-64373-1_4 <https://doi.org/10.1007/978-3-319-64373-1_4>`_

.. [des2019] Desain, P. W. M., Thielen, J., van den Broek, P. L. C., & Farquhar, J. D. R. (2019). U.S. Patent No. 10,314,508. Washington, DC: U.S. Patent and Trademark Office. `Link <https://patentimages.storage.googleapis.com/40/a3/bb/65db00c7de99ec/US10314508.pdf>`_

.. [ahm2019] Ahmadi, S., Borhanazad, M., Tump, D., Farquhar, J., & Desain, P. (2019). Low channel count montages using sensor tying for VEP-based BCI. Journal of Neural Engineering, 16(6), 066038. DOI: `10.1088/1741-2552/ab4057 <https://doi.org/10.1088/1741-2552/ab4057>`_

.. [thi2021] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5), 056007. DOI: `10.1088/1741-2552/abecef <https://doi.org/10.1088/1741-2552/abecef>`_

.. [ver2021] Verbaarschot, C., Tump, D., Lutu, A., Borhanazad, M., Thielen, J., van den Broek, P., ... & Desain, P. (2021). A visual brain-computer interface as communication aid for patients with amyotrophic lateral sclerosis. Clinical Neurophysiology, 132(10), 2404-2415. DOI: `10.1016/j.clinph.2021.07.012 <https://doi.org/10.1016/j.clinph.2021.07.012>`_

.. [mar2021] Martínez-Cagigal, V., Thielen, J., Santamaría-Vázquez, E., Pérez-Velasco, S., Desain, P., & Hornero, R. (2021). Brain–computer interfaces based on code-modulated visual evoked potentials (c-VEP): a literature review. Journal of Neural Engineering. DOI: `10.1088/1741-2552/ac38cf <https://doi.org/10.1088/1741-2552/ac38cf>`_

.. [thi2023] Thielen, J. (2023). Effects of Stimulus Sequences on Brain-Computer Interfaces Using Code-Modulated Visual Evoked Potentials: An Offline Simulation. In International Work-Conference on Artificial Neural Networks (pp. 555-568). Cham: Springer Nature Switzerland. DOI: `10.1007/978-3-031-43078-7_45 <https://doi.org/10.1007/978-3-031-43078-7_45>`_

Datasets
--------

On the Radboud Data Repository (`RDR <https://data.ru.nl/>`_):

.. [thi2018rdr] Thielen et al. (2018) Broad-Band Visually Evoked Potentials: Re(con)volution in Brain-Computer Interfacing. DOI: `10.34973/1ecz-1232 <https://doi.org/10.34973/1ecz-1232>`_
.. [ahm2018rdr] Ahmadi et al. (2018) High density EEG measurement. DOI: `10.34973/psaf-mq72 <https://doi.org/10.34973/psaf-mq72>`_
.. [ahm2019rdr] Ahmadi et al. (2019) Sensor tying. DOI: `10.34973/ehq6-b836 <https://doi.org/10.34973/ehq6-b836>`_
.. [thi2021rdr] Thielen et al. (2021) From full calibration to zero training for a code-modulated visual evoked potentials brain computer interface. DOI: `10.34973/9txv-z787 <https://doi.org/10.34973/9txv-z787>`_

On Mother of all BCI Benchmarks (`MOABB <https://moabb.neurotechx.com/docs/index.html>`_):

.. [thi2021moabb] c-VEP dataset from Thielen et al. (2021). `Link <https://moabb.neurotechx.com/docs/generated/moabb.datasets.Thielen2021.html#moabb.datasets.Thielen2021>`_

Contact
-------

* Jordy Thielen (jordy.thielen@donders.ru.nl)

.. toctree::
   :glob:
   :hidden:
   :maxdepth: 10
   :caption: Contents
   :titlesonly:

   Getting Started <tutorials/index>
   Examples <examples/index>
   API <api>