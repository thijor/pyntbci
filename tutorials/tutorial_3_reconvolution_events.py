"""
Reconvolution events
====================

This tutorial shows the basic building block of the reconvolution approach (see [1]_ and [2]_) that is implemented in
the PyntBCI library for analysing code-modulated responses. This tutorial generates an arbitrary code (here an
m-sequence) and shows different kinds of event definitions.

References
----------
.. [1] Thielen, J., van den Broek, P., Farquhar, J., & Desain, P. (2015). Broad-Band visually evoked potentials:
       re(con)volution in brain-computer interfacing. PLOS ONE, 10(7), e0133797.
       DOI: https://doi.org/10.1371/journal.pone.0133797
.. [2] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a
       code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5),
       056007. DOI: https://doi.org/10.1088/1741-2552/abecef
"""

import matplotlib.pyplot as plt
import seaborn

import pyntbci

seaborn.set_context("paper", font_scale=1.5)

# %%
# The stimulus
# --------
# PyntBCI contains a `stimulus` module with several functions that create various well-known noise-codes. Here we
# generate an m-sequence.

# Generate an m-sequence
V = pyntbci.stimulus.make_m_sequence()
n_classes, n_samples = V.shape
print("V shape: ", V.shape, "(classes, samples)")

# %%
# The event matrix
# ----------------
# In reconvolution, sequences are decomposed into individual events. Reconvolution then learns a separate evoked
# response for each of these events. The event definition can be manually set. An event that is commonly used is the
# `dur` (duration) event, which defines an event for each run-length of ones in a sequence. There are many other
# potential events that can be used, such as `id` (identity), `on`, `off`, `onoff` (on and off), `re` (rising edge),
# `fe` (falling edge), and `refe` (rising and falling edge, contrast). Here, we show what these decompositions look like
# in more detail.

events = pyntbci.utilities.EVENTS  # the event definitions
i_class = 0  # the class to visualize

for event in events:
    print(event)

    # Create event matrix
    E, events = pyntbci.utilities.event_matrix(V, event=event, onset_event=True)
    print("\tE shape:", E.shape, "(classes x events x samples)")
    print("\tEvents:", ", ".join([str(event) for event in events]))

    # Visualize event time-series
    fig, ax = plt.subplots(1, 1, figsize=(15, 3))
    pyntbci.plotting.eventplot(V[i_class, :], E[i_class, :, :], fs=60, ax=ax, events=events)
    ax.set_title(f"Event time-series {event} (code {i_class})")
    plt.tight_layout()

# plt.show()
