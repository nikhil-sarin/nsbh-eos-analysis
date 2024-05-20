import bilby
import numpy as np
import sys
import pandas as pd
from bilby.core.prior import Constraint, Uniform

np.random.seed(1234)
data = pd.read_csv('gw_detectable_events.csv')
ejecta_mass = data['ejecta_mass']
m_eject_ref = 0.01
D_ref = 500
em_selection = ejecta_mass / m_eject_ref / (data['luminosity_distance'] / D_ref)**2 > 1.
mmevent_indices = data[em_selection].index
job = int(sys.argv[1]) - 1
event_id = mmevent_indices[job]
data = data.iloc[event_id]
index = mmevent_indices
if event_id not in index:
    print('Event ID {} not in MM event index'.format(event_id))
    exit(0)
else:
    print('Event ID {} in index'.format(event_id))
chirp_mass = data['chirp_mass']
injection_parameters = data.to_dict()
keep = ['mass_1', 'mass_2', 'chi_1', 'chi_2','luminosity_distance', 'theta_jn',
        'psi', 'phase', 'geocent_time', 'ra', 'dec', 'lambda_1', 'lambda_2']
injection_parameters = {k: injection_parameters[k] for k in keep}
injection_parameters['lambda_1'] = 0.
print(injection_parameters)
outdir = "single_events"
label = str(event_id)

print('Creating NSBH injection for ID = {}'.format(event_id))
duration = 160.
sampling_frequency = 2048.
minimum_frequency = 20.
start_time = injection_parameters['geocent_time'] + 2 - duration

waveform_arguments = dict(waveform_approximant='SEOBNRv4_ROM_NRTidalv2_NSBH',
                          reference_frequency=20., minimum_frequency=minimum_frequency)
waveform_generator = bilby.gw.WaveformGenerator(duration=duration,
                                                sampling_frequency=sampling_frequency,
                                                frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
                                                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
                                                waveform_arguments=waveform_arguments)

interferometers = bilby.gw.detector.InterferometerList(["H1", "L1", "A1", "V1", "K1"])
H1_freqs, H1_ASD = np.genfromtxt('noise_curves/Aplus_asd.txt', unpack=True)
v1_freqs, v1_ASD = np.genfromtxt('noise_curves/avirgo_O5high_NEW.txt', unpack=True)
k1_freqs, k1_ASD = np.genfromtxt('noise_curves/kagra_128Mpc.txt', unpack=True)

interferometers[0].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=H1_freqs, asd_array=H1_ASD)
interferometers[1].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=H1_freqs, asd_array=H1_ASD)
interferometers[2].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=H1_freqs, asd_array=H1_ASD)
interferometers[3].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=v1_freqs, asd_array=v1_ASD)
interferometers[4].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=k1_freqs, asd_array=k1_ASD)

for interferometer in interferometers:
    interferometer.minimum_frequency = minimum_frequency
    interferometer.maximum_frequency = 2048.

interferometers.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=start_time)
interferometers.inject_signal(parameters=injection_parameters,
                              waveform_generator=waveform_generator)

mf_snrs = []
opt_snrs = []
for ifo in interferometers:
    mf_snrs.append(np.abs(ifo.meta_data['matched_filter_SNR']))
    opt_snrs.append(ifo.meta_data['optimal_SNR'])
opt_snr = np.array(opt_snrs)
mf_snr = np.array(mf_snrs)
opt_snr = np.sqrt(np.sum(opt_snr ** 2))
mf_snr = np.sqrt(np.sum(mf_snr ** 2))
print('Network matched filter SNR: ', mf_snr)
print('Network optimal SNR: ', opt_snr)

priors = bilby.gw.prior.BNSPriorDict(aligned_spin=True)
priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(1.3, 8, name='chirp_mass', unit='$M_{\\odot}$')
priors['lambda_2'] = Uniform(0, 5000, name='lambda_2', latex_label='$\Lambda_2$')
priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(0.01, 0.8, name='mass_ratio')
priors['chi_1'] = bilby.gw.prior.AlignedSpin(name='chi_1', a_prior=Uniform(minimum=0., maximum=0.99))
priors['chi_2'] = bilby.gw.prior.AlignedSpin(name='chi_2', a_prior=Uniform(minimum=0., maximum=0.08))
priors['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(50, 2050, name='luminosity_distance')
priors['mass_1'] = Constraint(minimum=2.3, maximum=70, name='mass_1', latex_label='$m_1$', unit=None)
priors['mass_2'] = Constraint(minimum=1.0, maximum=3, name='mass_2', latex_label='$m_2$', unit=None)

for key in ['lambda_1', 'geocent_time', 'phase']:
    priors[key] = bilby.core.prior.DeltaFunction(injection_parameters[key])

# Perform a check that the prior does not extend to a parameter space longer than the data
priors.validate_prior(duration, minimum_frequency=minimum_frequency)
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=interferometers,
    waveform_generator=waveform_generator,
    priors=priors, distance_marginalization=True)

m1 = injection_parameters["mass_1"]
m2 = injection_parameters["mass_2"]
injection_parameters["chirp_mass"] = bilby.gw.conversion.component_masses_to_chirp_mass(m1, m2)
injection_parameters["mass_ratio"] = m2 / m1

# Run sampler.
nwalkers=200
start_pos = bilby.core.prior.PriorDict()
for key in ['chi_1', 'chi_2', 'mass_ratio', 'psi']:
    start_pos[key] = bilby.core.prior.Normal(injection_parameters[key], 1e-3)
for key in ['ra', 'dec']:
    start_pos[key] = bilby.core.prior.Normal(injection_parameters[key], 1e-3)
for key in ['lambda_2']:
    start_pos[key] = bilby.core.prior.TruncatedNormal(injection_parameters[key], 100, minimum=0, maximum=5000)
for key in ['theta_jn']:
    start_pos[key] = bilby.core.prior.Normal(injection_parameters[key], 0.01)
for key in ['chirp_mass']:
    start_pos[key] = bilby.core.prior.Normal(injection_parameters[key], 1e-4)
pos0 = pd.DataFrame(start_pos.sample(nwalkers))

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    nwalkers=nwalkers,
    nsteps=2000,
    nburn=1700,
    sampler="emcee",
    pos0=pos0,
    nsamples=500,
    L1steps=80,
    L2steps=20,
    proposal_cycle='gwA',
    ntemps=1,
    thin_by_nact=0.2,
    clean=False,
    resume=True,
    Tmax_from_SNR=88,
    num_repeats=500,
    walks=20,
    nact=3,
    maxmcmc=10000,
    injection_parameters=injection_parameters,
    outdir=outdir,
    analytic_priors=True,
    checkpoint_every = 5000,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
)
# result.plot_corner()