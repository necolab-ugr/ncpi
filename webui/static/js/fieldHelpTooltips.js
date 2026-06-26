(() => {
    const enabledPrefixes = ["/simulation", "/field_potential", "/features", "/inference", "/analysis"];
    if (!enabledPrefixes.some((prefix) => window.location.pathname.startsWith(prefix))) return;

    const helpByKey = {
        tstop: "Total simulated duration in ms. Increase it to observe longer dynamics, at the cost of proportionally longer computation and larger outputs.",
        dt: "Simulation integration time step in ms. Smaller values improve temporal resolution and numerical accuracy but increase runtime and output size.",
        local_num_threads: "Number of CPU threads used by the simulator. Use a positive integer appropriate for the available CPU cores; excessive values can reduce performance.",
        areas: "Area labels used across all area-indexed parameters. Changing them here updates the labels shown in all other parameters and the area selector.",
        x: "Population labels used across all population-indexed parameters. Changing them here updates the labels shown in all other parameters.",
        n_x: "Number of neurons in each population. Use positive integers; larger populations increase runtime and memory use.",
        c_m_x: "Membrane capacitance for each population, in pF. Use positive values; larger values make membrane voltage respond more slowly to current.",
        tau_m_x: "Membrane time constant for each population, in ms. Use positive values; larger values make membrane voltage decay more slowly.",
        e_l_x: "Resting or leak reversal potential for each population, in mV. It defines the voltage approached without synaptic or injected input.",
        model: "Neuron model fixed by the selected simulation template. Read-only values cannot be changed from this form.",
        c_yx: "Connection-probability matrix from source populations X to target populations Y. Use values from 0 to 1; larger values create denser recurrent connectivity.",
        j_yx: "Synaptic-weight matrix from source populations X to target populations Y, in nA. Sign determines excitatory or inhibitory effect, and larger absolute values strengthen connections.",
        delay_yx: "Synaptic transmission-delay matrix from source populations X to target populations Y, in ms. Use non-negative values; larger values make activity arrive later.",
        tau_syn_yx: "Synaptic-current decay time constants for each source-target connection type, in ms. Use positive values; larger values produce longer-lasting postsynaptic effects.",
        n_ext: "Effective number of external input connections received by each neuron. Use non-negative integers; larger values increase external input.",
        nu_ext: "Rate of each external Poisson input, in Hz. Use a non-negative value; larger values increase external input activity and generally raise network firing.",
        j_ext: "Strength of each external synaptic input, in nA. Sign determines excitatory or inhibitory effect, and larger absolute values strengthen the external input.",
        p: "Connection probability. Use a value between 0 and 1, where larger values create denser connectivity.",
        extent: "Spatial size of the modeled network. Configure it using the units expected by the selected simulation model.",
        ou_sigma: "Standard deviation of the Ornstein-Uhlenbeck noise process. Larger values produce stronger fluctuations in the external drive.",
        ou_tau: "Correlation time of the Ornstein-Uhlenbeck noise in ms. Larger values make random fluctuations change more slowly.",
        v_th_x: "Spike threshold voltage for each population. A neuron emits a spike when its membrane voltage reaches this level.",
        v_reset_x: "Membrane voltage assigned immediately after a spike. Provide one reset potential per population.",
        t_ref_x: "Absolute refractory period after a spike, in ms. During this time the neuron cannot emit another spike.",
        g_l_x: "Leak conductance for each population. Larger values pull membrane voltage toward the leak reversal potential more strongly.",
        e_ex_x: "Excitatory synaptic reversal potential in mV. Keep it above typical membrane potentials for excitatory currents.",
        e_in_x: "Inhibitory synaptic reversal potential in mV. Keep it below typical membrane potentials for inhibitory currents.",
        tau_rise_ampa_x: "AMPA conductance rise time in ms. It controls how quickly excitatory synaptic responses reach their peak.",
        tau_decay_ampa_x: "AMPA conductance decay time in ms. It controls how long excitatory synaptic responses persist.",
        tau_rise_gaba_a_x: "GABA-A conductance rise time in ms. It controls how quickly inhibitory synaptic responses reach their peak.",
        tau_decay_gaba_a_x: "GABA-A conductance decay time in ms. It controls how long inhibitory synaptic responses persist.",
        i_e_x: "Constant current injected into each population. Its sign and magnitude shift baseline excitability.",
        sim_run_mode: "Single trial runs one simulation using the parameter values shown in the form. Parameter grid sweep runs multiple simulations while exploring selected parameters. In parameter exploration, ungrouped parameters form a Cartesian grid search; joint sweep groups link selected parameters by candidate index so they change together instead. For each swept parameter, Start is the first simulated value, Step is the amount added between values, and End is the target limit. Use a positive Step when End is above Start and a negative Step when End is below Start. End is included only when repeated Step increments reach it exactly.",
        four_area_local_editor: "Configure the model's four brain-area names and the neuronal-population names shared by every area. Then use the Area selector to choose which area's local Network and recurrent connectivity parameters are displayed and edited. Simulation parameters and inter-area connectivity remain shared across all areas.",
        sim_numpy_seed: "Seed used for NumPy randomization. Enable Fix seed and enter a non-negative integer to reproduce stochastic parameter generation and repeated runs. Leave Fix seed disabled to use non-fixed randomization.",
        grid_start: "Starting value simulated for this parameter.",
        grid_step: "Amount added to the parameter between simulations. Use a positive step when End is above Start and a negative step when End is below Start.",
        grid_end: "Final target value for the parameter exploration. It is included only when repeated Step increments land on it; otherwise the last simulated value is the final stepped value before End.",
        sim_repetitions: "Number of times each simulation configuration is run. Use multiple repetitions to characterize stochastic variability.",

        proxy_method: "Field-potential proxy formula to compute from simulation outputs. Available methods require different combinations of spikes, voltages, or synaptic currents.",
        sim_step: "Sampling interval of the simulation inputs. It is used to construct the output time axis and must match the source data.",
        proxy_decimation_factor: "Integer downsampling factor applied to the proxy output. Larger values reduce temporal resolution and output size.",
        bin_size: "Time-bin width used when converting spikes into firing rates. Larger bins smooth activity more strongly.",
        excitatory_only: "When enabled, calculate the selected proxy using only the excitatory population.",
        dt_kernel: "Temporal resolution used for kernel computation. Smaller values improve resolution but increase runtime and memory use.",
        t_x: "Time in ms at which presynaptic populations are activated when generating kernels. It defines the activation onset used to align the kernel response. Leave empty to use KernelParams.transient from the selected kernel parameters module.",
        tau: "Kernel time lag in ms defining the duration or lag window represented by the generated kernels. Larger values retain responses over a longer interval but increase computation and output size. Leave empty to use KernelParams.tau from the selected kernel parameters module.",
        mean_nu_x: "Mean firing rate assumed for each presynaptic population during kernel construction.",
        vrest_value: "Resting membrane potential used by the biophysical kernel calculation.",
        g_eff: "Include effective conductance corrections when constructing kernels. Enable when the selected kernel model supports this approximation.",
        kernel_population_size_e: "Number of excitatory neurons represented by the spike data. Use a non-negative integer matching the simulation.",
        kernel_population_size_i: "Number of inhibitory neurons represented by the spike data. Use a non-negative integer matching the simulation.",
        cdm_component: "Select which spatial component of dipole kernels is retained during convolution. Choose z to compute only the z-oriented current dipole moment, which is commonly used for vertically aligned cortical populations, or xyz (all) to preserve all three Cartesian components for later vector analysis or M/EEG projection. This setting applies only to dipole probes; scalar LFP probes such as GaussCylinderPotential retain all electrode channels.",
        cdm_mode: "Controls how the finite spike-rate signal and kernel overlap during convolution. same returns the centered portion with the length of the longer input and is usually appropriate for direct comparison with the simulated time series. full includes every partial overlap and therefore extends the output at both ends. valid keeps only positions where the signals overlap completely, producing a shorter output without edge contributions.",
        cdm_scale: "Multiplicative factor applied after convolution to set the numerical scaling of every computed CDM/LFP signal. The default dt * 10^-3 converts the kernel sampling interval from milliseconds to seconds so firing rates expressed in Hz are integrated consistently over time. Change it only when a different normalization or unit conversion is required.",
        cdm_decimation_factor: "Integer downsampling factor applied after CDM/LFP computation. A value of N keeps one sample every N time points, reducing output size and sampling frequency by that factor while preserving the represented duration. Use 1 to disable decimation; larger values reduce temporal resolution and may discard fast signal changes.",
        meeg_model: "Select the physical volume-conductor model used to transform current dipole moments into EEG potentials or MEG fields. NYHeadModel computes EEG with a realistic New York head model and its built-in cortical and 10-20-aligned electrode geometry; it automatically controls locations and forward mode and is unavailable for four-area simulations. FourSphereVolumeConductor computes EEG in four concentric spherical tissue layers. InfiniteVolumeConductor computes EEG in an unbounded homogeneous conductor using each sensor's position relative to the dipole. InfiniteHomogeneousVolCondMEG computes MEG in an unbounded homogeneous conductor, while SphericallySymmetricVolCondMEG computes MEG with spherical symmetry. The selected model determines whether the output is EEG or MEG, which locations are editable, and the default dipole and sensor geometry.",
        meeg_forward_mode: "Controls how multiple dipoles contribute to the configured sensors. Simulate all dipoles simultaneously projects every dipole to every sensor and sums their contributions; ordinary Hagen and Cavallari inputs replicate the same CDM across configured dipoles, while four-area inputs use fixed frontal, parietal, temporal, and occipital dipoles with their corresponding area CDMs. Per-sensor independent simulation computes sensor i only from dipole i, requiring equal numbers of explicit sensors and dipoles for non-NYHead models. NYHeadModel always uses per-sensor independent mode automatically, and four-area simulations always use simultaneous mode.",

        features_n_jobs: "Number of worker processes used to compute features across dataframe rows. Use 1 for sequential execution and easier debugging. Larger values can reduce runtime when many signals are processed, but each worker consumes CPU and memory; do not exceed the resources available to the runtime.",
        features_chunksize: "Optional number of dataframe rows sent to each worker per multiprocessing task. Small chunks improve load balancing when signals vary in cost, while larger chunks reduce scheduling overhead. Leave empty to let the multiprocessing backend choose.",
        features_start_method: "Process-creation strategy used for parallel feature computation. spawn starts clean worker processes and is the most portable option. fork is faster on supported Unix systems but can inherit unsafe library state. forkserver creates workers from a dedicated server process and is available only on supported platforms.",
        features_subsample_percent: "Percentage of parsed dataframe rows retained before feature computation. Use 100 to process every row. Values below 100 select a deterministic random subset, which is useful for quick tests but produces an intentionally incomplete feature dataframe.",

        catch22_normalize: "Standardize each signal before computing the 22 catch22 descriptors by removing its mean and scaling its variance. Enable this when descriptors should be less sensitive to absolute offset and amplitude. Leave disabled when signal magnitude is scientifically meaningful.",

        specparam_fs: "Sampling frequency in Hz used to build the Welch power spectrum and frequency axis. Leave empty only when the parsed dataframe already contains a valid sampling frequency. An incorrect value shifts all reported peak frequencies and frequency-range boundaries.",
        specparam_select_peak: "Controls which fitted oscillatory peak is exposed through the selected peak outputs. max_pw chooses the peak with greatest power, max_cf_in_range chooses the highest center-frequency peak inside the fitted range, and all preserves all fitted peak values where supported.",
        specparam_freq_min: "Lower frequency bound in Hz of the spectrum passed to the spectral model. Frequencies below this value are excluded from fitting. Choose a bound above 0 and below Frequency range max.",
        specparam_freq_max: "Upper frequency bound in Hz of the spectrum passed to the spectral model. It must exceed Frequency range min and remain below the Nyquist frequency, which is half the sampling frequency.",
        specparam_welch_nperseg: "Number of time samples in each Welch segment. Larger segments improve frequency resolution but provide fewer segments to average and require sufficiently long signals. The default shown by the WebUI is derived from the detected sampling frequency when available.",
        specparam_welch_noverlap: "Number of samples shared by consecutive Welch segments. Increasing overlap provides more spectral averages but increases computation and correlation between segments. It must be smaller than Welch nperseg.",
        specparam_welch_nfft: "FFT length used for every Welch segment. It must be at least as large as the segment length. Values above Welch nperseg zero-pad the segment and create a denser frequency grid, but do not add true spectral resolution.",
        specparam_welch_window: "Window function applied to every Welch segment before the FFT to control spectral leakage. Hann is a common general-purpose choice. Other windows trade main-lobe width, amplitude accuracy, and side-lobe suppression.",
        specparam_welch_detrend: "Detrending applied independently to every Welch segment before spectral estimation. constant removes each segment mean, linear removes a fitted linear trend, and none preserves the segment unchanged.",
        specparam_welch_scaling: "Controls the units of the Welch result. density returns power spectral density per Hz, while spectrum returns power per frequency bin. Use density for comparisons that should remain stable across frequency resolution.",
        specparam_welch_average: "Method used to combine periodograms from Welch segments. mean is efficient for stable data; median is more robust to transient high-power segments and outliers.",
        specparam_model_peak_threshold: "Relative peak-detection threshold, expressed in standard deviations above the modeled aperiodic background. Larger values accept only more prominent oscillatory peaks and generally reduce the number of fitted peaks.",
        specparam_model_min_peak_height: "Minimum absolute height above the aperiodic fit required for a candidate oscillatory peak. Use it together with Peak threshold to suppress weak fitted peaks; 0 disables an additional absolute-height restriction.",
        specparam_model_max_n_peaks: "Maximum number of oscillatory peaks fitted within the selected frequency range. Use 0 to prevent peak fitting. Larger values permit more complex spectra but can increase overfitting.",
        specparam_model_peak_width_min: "Minimum accepted oscillatory peak bandwidth in Hz. Narrower candidate peaks are rejected. It must be positive and lower than Peak width max.",
        specparam_model_peak_width_max: "Maximum accepted oscillatory peak bandwidth in Hz. Broader candidate peaks are rejected. It must exceed Peak width min and should be appropriate for the fitted frequency range.",
        specparam_model_aperiodic_mode: "Shape used for the aperiodic spectral component. fixed fits offset and exponent as a straight line in log-log space. knee adds a bend parameter and is appropriate when the fitted spectrum visibly changes slope.",
        specparam_model_verbose: "Enable or disable messages produced by the spectral-model fitting backend. Enable for diagnosis of fitting problems; disable for routine batch processing.",
        specparam_model_metric_gof_adjrsquared: "Request adjusted R-squared as an additional model-fit metric. It measures explained variance while accounting for model complexity and can be selected for quality thresholds or saved output.",
        specparam_model_metric_error_mse: "Request mean squared error as an additional model-fit metric. Lower values indicate smaller residual error and the metric can be used for quality thresholds or saved output.",
        specparam_metric_policy: "Controls how fits that fail configured quality thresholds are represented. reject marks failing fits as invalid for downstream filtering, while flag retains the result and records its validity status for later review.",
        specparam_threshold_gof_rsquared: "Minimum accepted model R-squared from 0 to 1. Fits below this threshold fail the quality policy. Increase it for stricter goodness-of-fit requirements.",
        specparam_threshold_metric_1_name: "Optional additional fit metric used as a quality threshold. Select adjusted R-squared or mean squared error, then provide its limit in Additional threshold value. The selected metric must also be computed by the model.",
        specparam_threshold_metric_1_value: "Threshold applied to the selected additional fit metric. Its interpretation follows the metric: adjusted R-squared generally requires a minimum value, while mean squared error generally requires a maximum value.",
        specparam_output_key: "Select the value stored in the output dataframe Features column. Choose the full dictionary to preserve all fitted results, a named scalar for analysis or model training, or Custom path to extract another nested result.",
        specparam_output_path: "Nested dictionary path extracted when Saved output value is Custom path, for example metrics.gof_rsquared or aperiodic_params.1. The resolved value should be numeric; missing paths produce missing feature values.",
        specparam_debug: "Enable additional diagnostic behavior and messages during specparam computation. Use it to investigate failed or unexpected fits; leave disabled for normal batch processing.",
        specparam_normalize: "Standardize each input signal before Welch spectral estimation. This reduces differences caused by absolute signal amplitude while preserving relative spectral shape. Disable it when absolute power or amplitude-dependent spectral values are important.",
        specparam_drop_invalid_rows: "Remove output dataframe rows whose spectral fit is invalid or whose selected output value is missing. Disable this to preserve row alignment and inspect invalid results explicitly.",
        specparam_reject_nonpositive_exponent: "Treat zero or negative aperiodic exponents as invalid. When the selected output is aperiodic exponent, invalid values become missing; with row dropping enabled, affected rows are removed.",

        dfa_fs: "Sampling frequency in Hz used to convert DFA windows, trimming, and filter ranges into samples. Leave empty only when the parsed dataframe contains a valid sampling frequency.",
        dfa_normalize: "Standardize each signal before DFA preprocessing. Enable to reduce sensitivity to absolute offset and scale; disable when amplitude differences should remain part of the analysis.",
        dfa_input_is_envelope: "Describe the supplied signal type. Select false for raw time series that must be band-pass filtered and Hilbert-transformed into amplitude envelopes. Select true only when every input is already the intended band-limited amplitude envelope; filtering settings are then ignored.",
        dfa_trim_seconds: "Number of seconds removed from both edges after filtering and envelope extraction to reduce boundary artifacts. Use 0 to keep all samples. Excessive trimming can leave too little data for the requested DFA windows.",
        dfa_overlap: "Use 50% overlap between consecutive DFA windows at each evaluated scale. Overlap provides more fluctuation estimates but increases computation and dependence between windows.",
        dfa_fit_interval_min: "Smallest window duration in seconds included when fitting the log-log slope reported as DFA. It must be below Fit interval max and, when a compute interval is provided, lie inside it.",
        dfa_fit_interval_max: "Largest window duration in seconds included when fitting the log-log slope reported as DFA. It must exceed Fit interval min, lie inside the optional compute interval, and remain feasible for the signal length.",
        dfa_compute_interval_min: "Optional smallest DFA window duration in seconds evaluated by the algorithm. It may extend below the fit interval to retain diagnostic fluctuation values. Leave the compute pair empty to use method defaults.",
        dfa_compute_interval_max: "Optional largest DFA window duration in seconds evaluated by the algorithm. It must exceed Compute interval min and contain the complete fit interval.",
        dfa_frequency_min: "Lower cutoff in Hz of the single band used to filter raw signals before DFA. Provide it with Band frequency max. This single-band mode cannot be combined with Spectrum range and is ignored for envelope inputs.",
        dfa_frequency_max: "Upper cutoff in Hz of the single band used to filter raw signals before DFA. It must exceed Band frequency min and remain below Nyquist.",
        dfa_spectrum_min: "Lower bound in Hz used to generate multiple analysis bands across a spectrum range. Provide it with Spectrum range max. This mode cannot be combined with Band frequency and is ignored for envelope inputs.",
        dfa_spectrum_max: "Upper bound in Hz used to generate multiple analysis bands across a spectrum range. It must exceed Spectrum range min and remain within the supported frequency range.",
        dfa_bad_idxes: "Optional comma-separated zero-based channel indexes excluded before DFA computation, for example 0,2,5. Use indexes matching the channel order of the input sample.",
        dfa_hilbert_n_fft: "Optional FFT length used by the Hilbert-transform step when converting filtered raw signals into amplitude envelopes. Leave empty for the backend default. It is ignored when Input is envelope is true.",
        dfa_filter_kwargs: "Optional Python dictionary of keyword arguments passed to the filtering operation before Hilbert envelope extraction, for example {\"fir_design\":\"firwin\"}. Use valid backend filter arguments only. It is ignored for envelope inputs.",
        dfa_output_key: "Select the DFA result stored in the Features column. DFA value stores the fitted scaling exponent, intercept stores the fitted log-log intercept, full dictionary preserves diagnostics, and Custom path extracts another result.",
        dfa_output_path: "Dictionary path extracted when Saved output value is Custom path, for example DFA or dfa_intercept. The resolved value is converted to a numeric scalar for the Features column.",

        fei_fs: "Sampling frequency in Hz used for fEI windows, DFA scales, trimming, and filtering. Leave empty only when a valid sampling frequency is present in the parsed dataframe.",
        fei_normalize: "Standardize each signal before fEI preprocessing. Enable to reduce differences caused by absolute offset and scale; disable when amplitude magnitude must remain meaningful.",
        fei_input_is_envelope: "Describe the supplied signal type. Select false for raw signals that must be filtered and Hilbert-transformed. Select true only when every input already contains the intended band-limited amplitude envelope; frequency and filtering settings are then ignored.",
        fei_trim_seconds: "Number of seconds removed from both edges after filtering and envelope extraction to reduce boundary artifacts. Use 0 to disable trimming. Ensure enough data remains for fEI and DFA windows.",
        fei_window_size_sec: "Required duration in seconds of each amplitude window used to estimate fEI. Larger windows provide more stable local amplitude and fluctuation estimates but reduce the number of windows and temporal detail.",
        fei_window_overlap: "Fraction of each fEI amplitude window shared with the next window. Use a value from 0 inclusive to 1 exclusive: 0 creates non-overlapping windows, while larger values increase window count and computation.",
        fei_dfa_threshold: "Minimum DFA exponent used to consider the fEI estimate interpretable under the method’s scale-free criterion. Windows or results below the threshold are handled according to the fEI implementation and reported diagnostics.",
        fei_dfa_overlap: "Use 50% overlap between consecutive windows used by the internal DFA calculation. Overlap increases the number of fluctuation estimates at each scale but also increases computation.",
        fei_dfa_fit_interval_min: "Smallest DFA window duration in seconds used to fit the scaling exponent calculated alongside fEI. It must be below DFA fit interval max and lie inside the optional DFA compute interval.",
        fei_dfa_fit_interval_max: "Largest DFA window duration in seconds used to fit the scaling exponent calculated alongside fEI. It must exceed the minimum, fit inside the optional compute interval, and be feasible for the signal length.",
        fei_dfa_compute_interval_min: "Optional smallest window duration in seconds evaluated by the internal DFA calculation. It may extend below the fit interval to preserve diagnostic fluctuation values.",
        fei_dfa_compute_interval_max: "Optional largest window duration in seconds evaluated by the internal DFA calculation. It must exceed the minimum and contain the complete DFA fit interval.",
        fei_frequency_min: "Lower cutoff in Hz of the single band used to filter raw signals before fEI. Provide it with Band frequency max. It cannot be combined with Spectrum range and is ignored for envelope inputs.",
        fei_frequency_max: "Upper cutoff in Hz of the single band used to filter raw signals before fEI. It must exceed Band frequency min and remain below Nyquist.",
        fei_spectrum_min: "Lower bound in Hz used to generate multiple analysis bands for fEI. Provide it with Spectrum range max. It cannot be combined with Band frequency and is ignored for envelope inputs.",
        fei_spectrum_max: "Upper bound in Hz used to generate multiple analysis bands for fEI. It must exceed Spectrum range min and remain within the supported frequency range.",
        fei_bad_idxes: "Optional comma-separated zero-based channel indexes excluded before fEI computation, for example 0,2,5. Use indexes matching the channel order of the input sample.",
        fei_hilbert_n_fft: "Optional FFT length used by the Hilbert-transform step when converting filtered raw signals into amplitude envelopes. Leave empty for the backend default. It is ignored for envelope inputs.",
        fei_filter_kwargs: "Optional Python dictionary of keyword arguments passed to filtering before Hilbert envelope extraction, for example {\"fir_design\":\"firwin\"}. Use valid backend filter arguments only. It is ignored for envelope inputs.",
        fei_output_key: "Select the fEI result stored in the Features column. Options include the primary scalar, outlier-aware variants, DFA, diagnostic arrays, outlier count, the full result dictionary, or a custom nested path.",
        fei_output_path: "Dictionary path extracted when Saved output value is Custom path, for example fEI_outliers_removed, DFA, or wAmp. Scalar outputs are stored directly; diagnostic arrays are preserved when selected.",

        custom_normalize: "Standardize each signal before passing it to custom_feature(sample, params). Enable when the custom feature should be independent of absolute offset and scale. The function receives the normalized sample only when this option is selected.",
        custom_feature_script: "Python source code defining custom_feature(sample, params). The function runs once per parsed signal and must return a numeric scalar or numeric array; arrays must have equal length for every sample. Raise clear errors for invalid inputs and avoid relying on state that is unavailable in worker processes.",
        parser_data_locator: "Select the field, dataframe column, nested object path, or detected source value that contains the electrophysiological signal to parse. The resolved value should be a numeric time series or array. For arrays, use the axis mapping controls to identify channels, samples, IDs, and existing trials/epochs. This value becomes the canonical data column used by feature extraction.",
        parser_array_axes_enabled: "Enable explicit array-axis mapping when automatic orientation detection is uncertain or incorrect. The mapping tells the parser what each dimension of the selected data array represents and overrides heuristic axis detection. Assign each role to a different dimension and verify the selected dimensions against the inspected array shape.",
        parser_axis_channels: "Select the array dimension containing sensors or channels. The parser splits this dimension into separate canonical sensor rows. Choose None only when the data has no channel dimension or already represents one signal per parsed item.",
        parser_axis_samples: "Select the array dimension containing consecutive time samples. This is the signal dimension used for temporal segmentation, z-scoring, epoching, and feature computation. It must identify the time-series axis rather than channels, subjects, or trials.",
        parser_axis_ids: "Optionally select the dimension containing independent IDs or subjects. The parser treats entries along this axis as distinct observations. Choose None when IDs are supplied through metadata, filenames, folders, or are not represented as an array dimension.",
        parser_axis_epochs: "Optionally select a dimension containing trials or epochs already present in the source array. Entries along this axis receive separate epoch identifiers. Parser-driven epoching can still be enabled later to divide each existing trial into shorter windows.",
        parser_fs_source: "Choose how the parser obtains sampling frequency in Hz. Select Numeric value to apply one manually entered frequency to the parsed signals, choose a detected field or nested locator when frequency is stored in the source data, or choose None only when downstream operations do not require time-to-sample conversion. Correct sampling frequency is required for temporal segmentation, epoching, and frequency-based features.",
        parser_fs_manual: "Sampling frequency in Hz applied when Sampling frequency source is Numeric value. Enter samples per second, not the simulation time step. For example, a 0.0625 ms time step corresponds to 16000 Hz. The value must match the actual data to produce correct epoch lengths, time bounds, and spectral frequencies.",
        parser_ch_names_source: "Choose where channel names come from. Autocomplete creates generic sequential names, such as ch0, ch1, and ch2. Select a detected field when the dataset already stores channel names, or choose Manual list to enter comma-separated names below. Ensure the number and order of names match the channels in the data.",
        parser_sensor_names: "Comma-separated sensor or channel labels used when Channel names source is Manual list. Enter names in the same order as channels appear along the configured Channels axis, with one label per channel, for example Fz, Cz, Pz.",
        parser_recording_type_source: "Choose whether recording modality is assigned from one constant value, read from a detected source field, or left unset. Recording type becomes canonical metadata used to distinguish signals such as LFP, CDM, EEG, MEG, and ECoG in downstream feature and analysis workflows.",
        parser_recording_type: "Constant recording modality assigned to every parsed row when Recording type source uses a custom value. Select the physical signal type represented by the data; choose Unknown only when the modality cannot be determined.",
        parser_metadata_mode: "Choose how canonical metadata is populated. Empirical mode lets you map subject ID, group, species, and condition from source fields, filenames, token values, or constants. Simulated mode creates synthetic simulation metadata and uses available trial or repetition information for conditions.",
        parser_metadata_subject_id_source: "Select the source used to populate canonical subject_id for empirical data. Use a detected field or filename-derived value when each file contains different subjects, Custom value when all parsed rows belong to one subject, or None when subject identity is unavailable. Subject IDs are also required for joining subject-level additional metadata.",
        parser_metadata_subject_id: "Constant subject identifier assigned to every parsed row when Subject ID source is Custom value. Use a stable identifier that can also match the selected link field in any additional metadata table.",
        parser_metadata_group_source: "Select the source used to populate canonical group metadata, such as control, treatment, cohort, or diagnosis. The value may come from source data, filenames, filename-format tokens, or a constant. Choose None when no meaningful group classification exists.",
        parser_metadata_group: "Constant group label assigned to every parsed row when Group source is Custom value, for example control or treatment.",
        parser_metadata_species_source: "Select the source used to populate canonical species metadata. Use a source field, filename-derived value, or constant that describes the organism represented by each recording; choose None when species is unknown or irrelevant.",
        parser_metadata_species: "Constant species label assigned to every parsed row when Species source is Custom value, for example human, mouse, or rat.",
        parser_metadata_condition_source: "Select the source used to populate canonical experimental condition metadata, such as rest, task, baseline, or stimulation. Conditions can come from source fields, filenames, filename-format tokens, or a constant and are commonly used for grouping feature outputs.",
        parser_metadata_condition: "Constant experimental-condition label assigned to every parsed row when Condition source is Custom value, for example rest or baseline.",
        parser_subject_id_locator: "Select the source field or filename-derived value that becomes canonical subject_id for each empirical recording. This identifier distinguishes subjects and is used as the key when joining additional subject-level metadata.",
        parser_group_locator: "Select the source field or filename-derived value that becomes canonical group metadata for each empirical recording, such as cohort, treatment, or diagnosis.",
        parser_species_locator: "Select the source field or filename-derived value that becomes canonical species metadata for each empirical recording.",
        parser_condition_locator: "Select the source field or filename-derived value that becomes canonical condition metadata for each empirical recording, such as rest, task, baseline, or stimulation.",
        parser_zscore: "Apply z-score normalization before epoching by subtracting the mean and dividing by the population standard deviation on each continuous signal. This can be combined with after-epoch z-scoring to normalize twice.",
        parser_zscore_after_epoch: "Apply z-score normalization after epoching so every generated epoch is standardized independently. This can be combined with before-epoch z-scoring.",
        parser_segment_t0_s: "Optional non-negative start time in seconds for cropping each time-domain signal before normalization, epoching, and aggregation. Set only this value to keep data from the selected time to the end. Leave both segment times empty to use the full signal. The parser uses sampling frequency and existing time bounds to convert this time to samples.",
        parser_segment_t1_s: "Optional end time in seconds for cropping each time-domain signal before normalization, epoching, and aggregation. Set only this value to keep data from the beginning to the selected time. It must be greater than the start time when both are provided. Leave both segment times empty to use the full signal; when epoching is enabled, the selected segment must be at least one epoch long.",
        additional_file_link_field: "Select the column in the additional metadata table whose values match canonical subject_id in the parsed dataset. The parser uses this key to join subject-level annotations such as group, condition, species, and recording type. Choose a column with stable identifiers and compatible values; unmatched subjects retain their existing metadata.",
        parser_enable_epoching: "Split each continuous time-domain sensor row into fixed-length sliding windows before feature extraction. Epoching requires a valid sampling frequency. Existing trials or epochs remain distinct and can each be subdivided further. Complete windows are emitted and epoch counts are aligned across comparable series.",
        parser_enable_aggregate: "Combine parsed rows across selected categorical fields after segmentation, normalization, and epoching. Aggregation reduces multiple sensors, epochs, subjects, or metadata groups into fewer signals using the selected reduction method. Only aggregate signals that have compatible array shapes and meaningful units.",
        parser_aggregate_over: "Select one or more canonical fields to collapse during aggregation. For example, selecting sensor combines channels while preserving other metadata; selecting epoch combines epochs and removes epoch-specific time bounds. All non-selected metadata fields remain grouping keys.",
        parser_aggregate_method: "Reduction applied element-by-element to aligned signal arrays in each aggregation group. Sum preserves total contribution, mean produces the arithmetic average, and median provides a robust central signal. Input arrays within a group must have compatible shapes.",
        parser_aggregate_label_value: "Optional replacement label written into each aggregated field, such as all or aggregate, so output rows clearly indicate that multiple original categories were combined.",
        parser_epoch_length_s: "Duration in seconds of each fixed-length epoch generated from continuous time-domain signals. Sampling frequency converts this duration to samples. The value must be positive and cannot exceed the selected temporal segment; signals too short for one complete window remain unepoched.",
        parser_epoch_step_s: "Time shift in seconds between consecutive epoch starts. Use the same value as epoch length for adjacent non-overlapping windows, a smaller positive value for overlapping windows, or a larger value to leave gaps. When omitted, the parser uses the epoch length.",

        training_model_name: "Select the inverse model trained to predict neural-circuit parameters from feature vectors. Scikit-learn regressors produce point predictions and suit standard supervised regression. NPE, NLE, and NRE use simulation-based inference to estimate posterior information and require SBI prior bounds. Changing the model refreshes the example hyperparameters and parameter grid.",
        training_hyperparams_json: "Base model configuration provided as a valid JSON object. For scikit-learn models, use constructor arguments accepted by the selected estimator. For SBI models, configure estimator_kwargs, inference_kwargs, and build_posterior_kwargs. These values are used directly in Single trial mode and serve as the base configuration during hyperparameter search.",
        training_param_grid_json: "Candidate hyperparameter configurations evaluated in Hyperparameter search mode. Provide a JSON object whose values are candidate lists, or a JSON list of complete parameter objects. Every resulting configuration is evaluated with repeated cross-validation; larger grids multiply training time.",
        training_sbi_train_params_json: "Arguments passed to the SBI training procedure, such as max_num_epochs, training_batch_size, learning_rate, and show_train_summary. Use valid JSON keys supported by the installed SBI backend. These settings control optimization of each SBI density estimator.",
        training_sbi_eval_sampling_kwargs_json: "Posterior-sampling arguments used to evaluate SBI candidates during hyperparameter search. sample_shape controls the number of posterior samples generated per validation observation. NLE and NRE may also require MCMC-related settings such as method, thin, num_chains, or num_workers.",
        training_n_splits: "Number of cross-validation folds used for each candidate in Hyperparameter search mode. It must be greater than 1 and should not exceed the number of paired training rows. More folds train on more data per fold but increase runtime.",
        training_n_repeats: "Number of times the cross-validation procedure is repeated with different seeded splits in Hyperparameter search mode. More repeats provide a more stable estimate of model performance but multiply computation.",
        training_seed: "Random seed used for deterministic row subsampling, cross-validation splitting, and supported model randomization. Reuse the same seed and configuration to improve reproducibility.",
        training_scaler_type: "Transformation fitted only on training features when Use scaler is enabled and saved with the trained model. StandardScaler centers and scales variance; MinMaxScaler maps ranges; RobustScaler uses robust statistics; MaxAbsScaler scales by maximum absolute value.",
        training_use_scaler: "Fit and apply the selected feature scaler before model training, then save it with the trained assets for consistent prediction preprocessing. Scaling is usually important for neural networks, SVR, Ridge, and other magnitude-sensitive models, but often unnecessary for random forests.",
        training_sklearn_verbose: "Non-negative verbosity level passed to supported scikit-learn training and validation procedures. Use 0 for minimal output and larger values for more progress information. It does not apply to SBI models.",
        training_subsample_percent: "Percentage of paired feature and parameter rows randomly retained before training. Use 100 or leave empty to use the complete dataset. Subsampling uses Seed and preserves X/Y row pairing; lower percentages reduce runtime but discard training information.",
        sbi_prior: "Lower and upper bounds of the BoxUniform prior for each inferred theta parameter. Add one ordered pair per target dimension, ensure low is strictly below high, and choose bounds that cover plausible values and the simulation support.",
        features_source_mode: "Choose where feature data is loaded from. The selected data must match the feature structure expected by the trained model.",
        inference_model_assets_source: "Choose the source of trained model assets. Include the model and any scaler used during training.",
        inference_scaler: "Scaler fitted during model training. Supply it when training used scaling so prediction inputs receive the same transformation.",
        sbi_sample_shape: "Number or shape of posterior samples generated per observation. More samples improve posterior summaries but increase runtime.",
        inference_model_name: "Select the prediction backend associated with the uploaded model artifact. Auto is recommended because it inspects the artifact and initializes the matching scikit-learn or SBI model. Choose a model manually only when automatic detection fails and you know the artifact type; selecting an incompatible model causes loading or prediction errors.",
        use_scaler: "Apply the uploaded fitted scaler to feature vectors before prediction. Enable this only when the selected model was trained with that exact scaler. The option is unavailable until a scaler artifact is provided; omitting a required scaler or applying the wrong one produces predictions from incompatible inputs.",
        inference_n_jobs: "Number of worker processes used for scikit-learn prediction across feature rows. Use 1 for sequential execution and easier debugging. Larger values can reduce runtime for large inputs but consume additional CPU and memory. This setting does not apply to SBI prediction.",
        inference_chunksize: "Optional number of feature rows assigned to each scikit-learn worker task. Smaller chunks improve load balancing, while larger chunks reduce multiprocessing overhead. Leave empty to use backend defaults. This setting does not apply to SBI prediction.",
        inference_start_method: "Process-creation strategy used for parallel scikit-learn prediction. spawn is portable and starts clean workers; fork can be faster on supported Unix systems but inherits process state; forkserver uses a dedicated worker server where supported. This setting does not apply to SBI prediction.",
        inference_sbi_eval_sampling_kwargs_json: "Valid JSON arguments controlling posterior sampling for SBI prediction. sample_shape sets how many posterior samples are drawn per feature row. NLE and NRE may also use MCMC settings such as method, thin, num_chains, or num_workers. More samples improve summary stability but increase runtime and memory use.",
        sbi_summary_mode: "Statistic used to convert posterior samples for each feature row into the value stored in Predictions. Posterior mean averages samples and is sensitive to skew and outliers; posterior median is more robust and may better represent asymmetric posterior distributions.",
        inference_subsample_percent: "Percentage of input feature rows used for prediction. Use 100 to predict every row. Values below 100 select a deterministic random subset using seed 0, and the saved output contains predictions only for that retained subset.",

        boxplot_group_by: "Categorical column used to divide values into boxplot groups.",
        boxplot_value_col: "Numeric column summarized by the boxplot.",
        boxplot_showfliers: "Display observations beyond the whiskers as individual outlier markers.",
        boxplot_width: "Relative width of each box. Enter a positive numeric value.",
        boxplot_linewidth: "Width of boxplot outlines and whiskers.",
        boxplot_control_group: "Reference category used when computing effect sizes against other groups.",
        topomap_signal_type: "Select EEG or MEG so the appropriate sensor geometry and plotting conventions are used.",
        topomap_meg_atlas: "Atlas or sensor layout used to position MEG values on the topographic map.",
        topomap_grouping_mode: "Choose whether maps summarize each sensor grouping or compare categories within a selected column.",
        topomap_compare_method: "Method used to compare categories: raw subtraction preserves units, while Cohen's d expresses standardized effect size.",
        topomap_cohend_abs_threshold: "Minimum absolute Cohen's d displayed. Increase it to hide weak effects.",
        topomap_group_by: "Categorical column defining the groups or conditions displayed in topographic maps.",
        topomap_control_group: "Reference category used for category comparisons.",
        topomap_value_col: "Numeric measurement mapped to sensor positions.",
        topomap_sensor_col: "Column containing sensor or channel names. Names must match the selected montage or atlas.",
        topomap_eeg_montage: "EEG electrode montage used to obtain standard sensor coordinates.",
        topomap_eeg_ch_type: "MNE channel type used when constructing EEG sensor information.",
        topomap_cmap: "Matplotlib colormap used to encode values. Choose a sequential map for magnitudes or a diverging map for signed contrasts.",
        topomap_vmin: "Lower color-scale limit. Leave as auto to derive it from the plotted data.",
        topomap_vmax: "Upper color-scale limit. Leave as auto to derive it from the plotted data.",
        topomap_res: "Image resolution of the interpolated topographic map. Higher values improve smoothness but increase rendering time.",
        topomap_contours: "Number or mode of contour lines drawn over the topographic map.",
        topomap_extrapolate: "Controls interpolation outside the sensor convex hull.",
        topomap_image_interp: "Interpolation method used when rendering the topographic image.",
        topomap_show_sensors: "Show sensor locations on top of the interpolated map.",
        topomap_scale_mode: "Controls whether maps share a common color scale or use separate scales.",
        sim_plot_type: "Type of simulation result visualization to generate.",
        sim_trial_start: "First trial index included in the plot. Indices are zero-based.",
        sim_trial_end: "Last trial boundary included in the plot. Leave empty where supported to include remaining trials.",
        sim_time_start: "Beginning of the plotted time interval.",
        sim_time_end: "End of the plotted time interval.",
        sim_freq_min: "Lowest frequency included in spectral plots.",
        sim_freq_max: "Highest frequency included in spectral plots.",
        sim_cdm_psd_mode: "Method used to combine or display current-dipole-moment power spectra.",
        sim_welch_nperseg: "Number of samples per Welch segment used for simulation-result spectra.",
        sim_welch_noverlap: "Number of overlapping samples between adjacent Welch segments.",
        sim_welch_nfft: "FFT length used by Welch spectral estimation."
    };

    const cavallariHelpByKey = {
        p: "Probability of each recurrent source-target neuron pair being connected. Use a value from 0 to 1; larger values create denser connectivity and increase recurrent input.",
        exc_exc_recurrent: "Recurrent synaptic conductance from an excitatory source neuron to an excitatory target neuron, in nS. Use a positive value; larger values strengthen recurrent excitation of excitatory neurons.",
        exc_inh_recurrent: "Recurrent synaptic conductance from an excitatory source neuron to an inhibitory target neuron, in nS. Use a positive value; larger values strengthen excitation of inhibitory neurons.",
        inh_exc_recurrent: "Recurrent synaptic conductance from an inhibitory source neuron to an excitatory target neuron, in nS. Use a negative value; larger absolute values strengthen inhibition of excitatory neurons.",
        inh_inh_recurrent: "Recurrent synaptic conductance from an inhibitory source neuron to an inhibitory target neuron, in nS. Use a negative value; larger absolute values strengthen recurrent inhibition of inhibitory neurons.",
        th_exc_external: "Synaptic conductance from the thalamic external-input generator to excitatory neurons, in nS. Use a non-negative value; larger values strengthen thalamic drive to excitatory neurons.",
        th_inh_external: "Synaptic conductance from the thalamic external-input generator to inhibitory neurons, in nS. Use a non-negative value; larger values strengthen thalamic drive to inhibitory neurons.",
        cc_exc_external: "Synaptic conductance from the cortical external-input generator to excitatory neurons, in nS. Use a non-negative value; larger values strengthen fluctuating cortical input to excitatory neurons.",
        cc_inh_external: "Synaptic conductance from the cortical external-input generator to inhibitory neurons, in nS. Use a non-negative value; larger values strengthen fluctuating cortical input to inhibitory neurons.",
        v_0: "Constant baseline component of the thalamic external-input signal, in spikes/ms. Use a non-negative value; larger values increase the baseline thalamic input rate.",
        a_ext: "Amplitude of the sinusoidal component added to the thalamic external-input signal, in spikes/ms. Use a non-negative value; 0 disables the oscillatory component and larger values increase its modulation depth.",
        f_ext: "Frequency of the sinusoidal component added to the thalamic external-input signal, in Hz. Use a non-negative value; it has no effect when the oscillatory amplitude is 0.",
        ou_sigma: "Standard deviation of the Ornstein-Uhlenbeck process that generates cortical external-input fluctuations, in spikes/ms. Use a non-negative value; 0 disables fluctuations and larger values increase their magnitude.",
        ou_tau: "Time constant of the Ornstein-Uhlenbeck process that generates cortical external-input fluctuations, in ms. Use a positive value; larger values produce slower, more persistent fluctuations.",
        v_th_x: "Membrane-potential threshold at which each population's neurons emit a spike, in mV. It should be above the reset and leak reversal potentials; lower values make neurons easier to activate.",
        v_reset_x: "Membrane potential assigned after a spike for each population, in mV. It should be below the spike threshold; lower values generally delay the next spike.",
        t_ref_x: "Absolute refractory period after a spike for each population, in ms. Use a non-negative value; larger values reduce the maximum possible firing rate.",
        g_l_x: "Leak conductance for each population, in nS. Use a positive value; larger values pull membrane voltage toward the leak reversal potential more strongly.",
        e_ex_x: "Excitatory synaptic reversal potential for each population, in mV. It should normally be above resting and threshold potentials.",
        e_in_x: "Inhibitory synaptic reversal potential for each population, in mV. It should normally be below resting and threshold potentials.",
        tau_rise_ampa_x: "AMPA conductance rise time for each population, in ms. Use a positive value; larger values make excitatory conductance rise more slowly.",
        tau_decay_ampa_x: "AMPA conductance decay time for each population, in ms. Use a positive value greater than the AMPA rise time; larger values prolong excitatory input.",
        tau_rise_gaba_a_x: "GABA-A conductance rise time for each population, in ms. Use a positive value; larger values make inhibitory conductance rise more slowly.",
        tau_decay_gaba_a_x: "GABA-A conductance decay time for each population, in ms. Use a positive value greater than the GABA-A rise time; larger values prolong inhibitory input.",
        i_e_x: "Constant current injected into each population's neurons, in pA. Positive values depolarize neurons and negative values hyperpolarize them."
    };

    const fourAreaHelpByKey = {
        inter_area_c_yx: "Connection-probability matrix from source area-population nodes to target area-population nodes. Use values from 0 to 1; larger values create denser inter-area connectivity. Same-area entries are ignored (these connections are configured in the Recurrent connectivity section).",
        inter_area_j_yx: "Synaptic-weight matrix from source area-population nodes to target area-population nodes, in nA. Sign determines excitatory or inhibitory effect, and larger absolute values strengthen connections. Same-area entries are ignored (these connections are configured in the Recurrent connectivity section).",
        inter_area_delay_yx: "Transmission-delay matrix from source area-population nodes to target area-population nodes, in ms. Use non-negative values; larger values make activity arrive later. Same-area entries are ignored (these connections are configured in the Recurrent connectivity section)."
    };

    const pageHelp = [
        ["/simulation/upload_sim", "Load previously generated simulation pickle outputs, including complete simulation bundles or legacy spike-time, spike-gid, timing, and network files."],
        ["/simulation/new_sim/custom", "Configure a custom simulation by supplying the required model and parameter files. The uploaded code defines available parameters and output behavior."],
        ["/simulation/new_sim/hagen", "Configure the Hagen current-based excitatory/inhibitory LIF network, including population parameters and connectivity matrices."],
        ["/simulation/new_sim/four_area", "Configure four coupled cortical areas. Local parameters apply to the selected area; inter-area matrices configure connections between areas."],
        ["/simulation/new_sim/cavallari", "Configure the Cavallari conductance-based network, including spatial connectivity, external drive, synaptic conductances, and neuron dynamics."],
        ["/field_potential/load", "Load precomputed field-potential outputs such as current dipole moments, LFPs, M/EEG signals, or proxy signals."],
        ["/field_potential/kernel", "Configure biophysical kernels, convolve simulation spikes into current dipole moments or LFPs, and project dipoles to EEG or MEG sensors."],
        ["/field_potential/proxy", "Compute efficient field-potential proxy signals from simulation outputs such as spikes, membrane voltages, and synaptic currents."],
        ["/features/load_data", "Load precomputed feature dataframes"],
        ["/inference/load_data", "Load precomputed predictions"],
        ["/analysis", "Configure visualization and statistical analysis of dataframes or simulation outputs, including boxplots, topographic maps, and simulation-result plots."]
    ];

    const normalize = (value) => String(value || "")
        .trim()
        .replace(/\./g, "_")
        .replace(/-/g, "_")
        .replace(/\s+/g, "_")
        .replace(/[^a-zA-Z0-9_]/g, "")
        .toLowerCase();

    const controlKey = (control) => {
        if (control.dataset.gridRole) return `grid_${normalize(control.dataset.gridRole)}`;
        const raw = control.dataset.param || control.name || control.id || "";
        let key = normalize(raw);
        if (key === "sim_use_numpy_seed") key = "sim_numpy_seed";
        if (key === "dt" && control.closest("form")?.id === "kernel-compute-form") key = "dt_kernel";
        if (key.includes("prior") && key.includes("bound")) key = "sbi_prior";
        if (key.includes("scaler") && !key.includes("type") && !key.includes("use")) key = "inference_scaler";
        return key;
    };

    const lookupHelp = (key) => {
        if (
            window.location.pathname.startsWith("/simulation/new_sim/cavallari")
            && cavallariHelpByKey[key]
        ) return cavallariHelpByKey[key];
        if (
            window.location.pathname.startsWith("/simulation/new_sim/four_area")
            && fourAreaHelpByKey[key]
        ) return fourAreaHelpByKey[key];
        if (helpByKey[key]) return helpByKey[key];
        const suffixMatch = Object.keys(helpByKey)
            .filter((candidate) => key.endsWith(candidate))
            .sort((a, b) => b.length - a.length)[0];
        return suffixMatch ? helpByKey[suffixMatch] : "";
    };

    const fallbackHelp = (control, labelText, key) => {
        const type = String(control.type || control.tagName || "").toLowerCase();
        const parts = [`Configures ${labelText || key.replace(/_/g, " ")}.`];
        if (control.tagName === "SELECT") {
            parts.push("Choose the option that matches the input data and intended computation.");
        } else if (type === "checkbox" || type === "radio") {
            parts.push("Enable or select this option when the described behavior should be applied.");
        } else if (type === "number" || type === "range") {
            const constraints = [];
            if (control.min !== "") constraints.push(`minimum ${control.min}`);
            if (control.max !== "") constraints.push(`maximum ${control.max}`);
            if (control.step && control.step !== "any") constraints.push(`step ${control.step}`);
            parts.push(constraints.length ? `Enter a numeric value (${constraints.join(", ")}).` : "Enter a numeric value using units indicated by the label.");
        } else if (control.tagName === "TEXTAREA" || type === "text") {
            parts.push("Configure the value according to the selected model or computation.");
        }
        return parts.join(" ");
    };

    const labelForControl = (control) => {
        if (control.closest("label")) return control.closest("label");
        if (control.id) {
            try {
                const label = document.querySelector(`label[for="${CSS.escape(control.id)}"]`);
                if (label) return label;
            } catch (_error) {
                return null;
            }
        }
        let container = control.parentElement;
        for (let depth = 0; container && depth < 3 && container.tagName !== "FORM"; depth += 1) {
            const siblingLabel = Array.from(container.children)
                .find((child) => child.tagName === "LABEL" && !child.contains(control));
            if (siblingLabel) return siblingLabel;
            container = container.parentElement;
        }
        return null;
    };

    const titleTarget = (label, control) => {
        const directSpans = Array.from(label.children).filter((child) =>
            child.tagName === "SPAN"
            && !child.classList.contains("material-symbols-outlined")
            && !child.classList.contains("field-help")
        );
        return directSpans.find((span) => !span.contains(control)) || label;
    };

    const pathBrowserNeedles = [
        "filebrowser",
        "folderbrowser",
        "serverbrowser",
        "file-browser",
        "folder-browser",
        "directory-browser",
        "serverfilebrowser",
        "serverfolderbrowser",
        "serverdirbrowser",
        "server-file-browser",
        "server-folder-browser",
        "server-dir-browser",
        "serverfilebrowseropen",
        "serverfolderbrowseropen",
        "serverdirbrowseropen"
    ];

    const isInsidePathBrowser = (node) => {
        let current = node instanceof Element ? node : node?.parentElement || null;
        while (current && current !== document.body) {
            if (current.dataset.fieldHelpSkip === "1") return true;
            const values = [
                current.id,
                current.getAttribute("x-show"),
                current.getAttribute("x-for"),
                current.getAttribute("@click"),
                current.getAttribute("data-browser"),
                current.getAttribute("data-role")
            ].map((value) => String(value || "").toLowerCase());
            if (values.some((value) => pathBrowserNeedles.some((needle) => value.includes(needle)))) {
                return true;
            }
            current = current.parentElement;
        }
        return false;
    };

    const removePathBrowserHelpIcons = (root = document) => {
        root.querySelectorAll(".field-help").forEach((icon) => {
            if (isInsidePathBrowser(icon)) icon.remove();
        });
    };

    let sequence = 0;
    const appendHelpIcon = (target, labelText, help, className = "") => {
        // Do not add field-help icons on the Analysis module pages per request
        try {
            if (window.location && window.location.pathname && window.location.pathname.startsWith('/analysis')) return null;
        } catch (_e) {
            // ignore
        }
        if (isInsidePathBrowser(target)) return null;
        if (!target || !help || target.querySelector(":scope > .field-help")) return null;
        const tooltipId = `field-help-tooltip-${++sequence}`;
        const icon = document.createElement("span");
        icon.className = `field-help ${className}`.trim();
        icon.tabIndex = 0;
        icon.setAttribute("role", "button");
        icon.setAttribute("aria-label", `Information about ${labelText}`);
        icon.setAttribute("aria-describedby", tooltipId);
        icon.setAttribute("aria-expanded", "false");
        // icon.innerHTML = `<span class="material-symbols-outlined" aria-hidden="true" style="font-size:18px">info</span><span id="${tooltipId}" class="field-help-tooltip" role="tooltip"></span>`;
        icon.querySelector(".field-help-tooltip").textContent = help;
        const positionTooltip = () => {
            const tooltip = icon.querySelector(".field-help-tooltip");
            if (!tooltip) return;
            const viewportPadding = 12;
            const iconRect = icon.getBoundingClientRect();

            tooltip.style.position = "fixed";
            tooltip.style.right = "auto";
            tooltip.style.left = "0px";
            tooltip.style.top = "0px";
            tooltip.style.transform = "none";

            const tooltipRect = tooltip.getBoundingClientRect();
            const width = tooltipRect.width || Math.min(352, window.innerWidth - (viewportPadding * 2));
            const height = tooltipRect.height || 120;
            const maxLeft = Math.max(viewportPadding, window.innerWidth - width - viewportPadding);
            let left = iconRect.left + (iconRect.width / 2) - (width / 2);
            left = Math.min(Math.max(viewportPadding, left), maxLeft);

            let top = iconRect.bottom + 8;
            if (top + height > window.innerHeight - viewportPadding) {
                top = iconRect.top - height - 8;
            }
            top = Math.max(viewportPadding, top);

            tooltip.style.left = `${left}px`;
            tooltip.style.top = `${top}px`;
        };
        const closeTooltip = ({suppressCurrentHover = false} = {}) => {
            icon.classList.remove("is-open");
            icon.setAttribute("aria-expanded", "false");
            if (suppressCurrentHover) {
                icon.classList.add("is-click-closed");
            }
        };
        const openTooltip = () => {
            document.querySelectorAll(".field-help.is-open").forEach((otherIcon) => {
                if (otherIcon !== icon) {
                    otherIcon.classList.remove("is-open");
                    otherIcon.setAttribute("aria-expanded", "false");
                }
            });
            icon.classList.remove("is-click-closed");
            icon.classList.add("is-open");
            icon.setAttribute("aria-expanded", "true");
            positionTooltip();
        };
        icon.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            if (icon.classList.contains("is-open")) {
                closeTooltip({suppressCurrentHover: true});
                icon.blur();
                return;
            }
            openTooltip();
        });
        icon.addEventListener("mouseleave", () => {
            icon.classList.remove("is-click-closed");
        });
        icon.addEventListener("keydown", (event) => {
            if (event.key !== "Enter" && event.key !== " ") return;
            event.preventDefault();
            if (icon.classList.contains("is-open")) {
                closeTooltip({suppressCurrentHover: true});
            } else {
                openTooltip();
            }
        });
        icon.addEventListener("mouseenter", positionTooltip);
        icon.addEventListener("focus", positionTooltip);
        icon.addEventListener("focusout", () => {
            icon.classList.remove("is-click-closed");
        });
        window.addEventListener("resize", () => {
            if (icon.classList.contains("is-open") || document.activeElement === icon || icon.matches(":hover")) {
                positionTooltip();
            }
        });
        target.appendChild(icon);
        return icon;
    };

    const annotateControl = (control) => {
        if (
            control.dataset.fieldHelpProcessed === "1"
            || control.type === "hidden"
            || control.type === "file"
            || control.disabled && control.type === "hidden"
            || isInsidePathBrowser(control)
        ) return;

        const key = controlKey(control);
        if (key === "sim_run_mode" || normalize(control.id) === "sim_use_numpy_seed") {
            control.dataset.fieldHelpProcessed = "1";
            return;
        }
        if (control.dataset.gridRole) {
            if (window.location.pathname.startsWith("/simulation/new_sim/")) {
                control.dataset.fieldHelpProcessed = "1";
                return;
            }
            const target = control.parentElement?.querySelector(":scope > span");
            const help = lookupHelp(key);
            if (target && help) appendHelpIcon(target, target.textContent.trim(), help);
            control.dataset.fieldHelpProcessed = "1";
            return;
        }

        const label = labelForControl(control);
        if (!label || label.dataset.fieldHelpSkip === "1") return;
        const target = titleTarget(label, control);
        if (
            window.location.pathname.startsWith("/simulation/new_sim/")
            && target.matches("[data-param-leaf-label='1']")
        ) {
            control.dataset.fieldHelpProcessed = "1";
            return;
        }
        if (target.querySelector(":scope > .field-help")) {
            control.dataset.fieldHelpProcessed = "1";
            return;
        }

        const labelText = label.textContent.replace(/\s+/g, " ").trim();
        const customHelp = control.dataset.help || label.dataset.help || "";
        const help = customHelp || lookupHelp(key) || fallbackHelp(control, labelText, key);
        if (!help) return;

        appendHelpIcon(target, labelText || key, help);
        control.dataset.fieldHelpProcessed = "1";
    };

    const annotate = (root = document) => {
        root.querySelectorAll("input, select, textarea").forEach(annotateControl);
    };

    const annotateStaticHelp = (root = document) => {
        root.querySelectorAll("[data-field-help-static='1'][data-help]").forEach((target) => {
            appendHelpIcon(
                target,
                target.textContent.replace(/\s+/g, " ").trim(),
                target.dataset.help
            );
        });
    };

    document.addEventListener("click", () => {
        document.querySelectorAll(".field-help.is-open").forEach((icon) => {
            icon.classList.remove("is-open");
            icon.classList.remove("is-click-closed");
            icon.setAttribute("aria-expanded", "false");
        });
    });

    document.addEventListener("keydown", (event) => {
        if (event.key !== "Escape") return;
        document.querySelectorAll(".field-help.is-open").forEach((icon) => {
            icon.classList.remove("is-open");
            icon.classList.remove("is-click-closed");
            icon.setAttribute("aria-expanded", "false");
        });
    });

    const start = () => {
        const pageEntry = pageHelp
            .filter(([path]) => window.location.pathname.startsWith(path))
            .sort((a, b) => b[0].length - a[0].length)[0];
        const pageTitle = document.querySelector("main h1, h1, main .text-4xl.font-black, .text-4xl.font-black");
        if (
            pageEntry
            && pageTitle
            && ![
                "/simulation/new_sim/hagen",
                "/simulation/new_sim/cavallari",
                "/simulation/new_sim/four_area",
                "/field_potential/kernel"
            ].some((path) => window.location.pathname.startsWith(path))
        ) {
            appendHelpIcon(pageTitle, pageTitle.textContent.replace(/\s+/g, " ").trim(), pageEntry[1], "field-help-page");
        }
        if (window.location.pathname.startsWith("/simulation/new_sim/")) {
            const simulationModeHeading = Array.from(document.querySelectorAll("h2"))
                .find((heading) => heading.textContent.replace(/\s+/g, " ").trim() === "Simulation mode");
            if (simulationModeHeading) {
                appendHelpIcon(simulationModeHeading, "Simulation mode", helpByKey.sim_run_mode);
            }
        }
        if (window.location.pathname.startsWith("/simulation/new_sim/four_area")) {
            const localAreaEditorHeading = Array.from(document.querySelectorAll("h2"))
                .find((heading) => heading.textContent.replace(/\s+/g, " ").trim() === "Brain area editor");
            if (localAreaEditorHeading) {
                appendHelpIcon(localAreaEditorHeading, "Brain area editor", helpByKey.four_area_local_editor);
            }
        }
        annotateStaticHelp();
        annotate();
        removePathBrowserHelpIcons();
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => mutation.addedNodes.forEach((node) => {
                if (!(node instanceof Element)) return;
                if (node.matches("[data-field-help-static='1'][data-help]")) annotateStaticHelp(node.parentElement || node);
                annotateStaticHelp(node);
                if (node.matches("input, select, textarea")) annotateControl(node);
                annotate(node);
                removePathBrowserHelpIcons(node);
            }));
        });
        observer.observe(document.body, {childList: true, subtree: true});
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start, {once: true});
    } else {
        start();
    }
})();
