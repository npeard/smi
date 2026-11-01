"""Channel naming of the real-data HDF5 schema, shared by readers and writers."""

# Photodiode channel -> laser wavelength in microns. RP1_CH1 is the speaker
# drive voltage and is not a photodiode. The 675 nm laser is spelled 0.675
# here and 0.6748 in the simulator configs (smi.data.frozen matches the two to
# RIG_WAVELENGTH_TOL_UM).
PD_WAVELENGTHS_UM: dict[str, float] = {
    'RP1_CH2': 0.635,
    'RP2_CH1': 0.675,
    'RP2_CH2': 0.515,
}
DRIVE_KEY = 'RP1_CH1'
