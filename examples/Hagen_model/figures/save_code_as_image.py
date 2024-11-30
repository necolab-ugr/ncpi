import pygments
from pygments import lexers
from pygments import formatters
from PIL import Image
from io import BytesIO
import textwrap

# The Python code snippet
code = '''
import ncpi

# Build the LIF network model and simulate it
sim = ncpi.Simulation(param_folder='params',
                      python_folder='python',
                      output_folder='output')
sim.network('network.py', 'network_params.py')
sim.simulate('simulation.py', 'simulation_params.py')

# Compute the spatiotemporal kernel
potential = ncpi.FieldPotential(kernel=True)
H_YX = potential.create_kernel(MC_model_folder,
                               MC_output_path,
                               kernelParams,
                               biophys,
                               dt,
                               tstop, 
                               electrodeParameters, 
                               CDM=True)
                               
# Compute the CDMs
probe = 'KernelApproxCurrentDipoleMoment'
kernel = H_YX[f'{X}:{Y}'][probe][2, :] # z-axis
CDMs = np.convolve(LIF_spike_rates, kernel, 'same')

# Obtain features from simulation and empirical data
features = ncpi.Features(method='catch22')
sim_df = features.compute_features(CDMs)
emp_df = features.compute_features(emp_data)

# Train the neural network model using 10-fold CV
hyperparams = [{'hidden_layer_sizes': (25,25)},
               {'hidden_layer_sizes': (50,50)}] 
inference = ncpi.Inference(model='MLPRegressor')
inference.add_simulation_data(sim_df['Features'],
                              theta) # parameters
inference.train(param_grid=hyperparams,
                n_splits=10,
                n_repeats=1) 
                
# Predict the cortical circuit parameters
predictions = inference.predict(emp_df['Features'])

# Perform the LMER analysis
analysis = ncpi.Analysis()
analysis.lmer(predictions)                                                 
'''

# Wrap the code to 80 characters
wrapped_code = "\n".join(textwrap.fill(line, width=60) for line in code.splitlines())

# Use the Python lexer for syntax highlighting
lexer = lexers.get_lexer_by_name('python')

# Create a formatter that will generate the image
formatter = formatters.ImageFormatter(
    font_name='Courier New',
    font_size=28,
    line_numbers=False,
    background_color = 'white',
    text_color = 'black',
    image_pad=5,
    line_pad=2,
    syntax_highlighting_style='monokai'
)

# Generate the highlighted code as an image
image_data = pygments.highlight(wrapped_code, lexer, formatter)

# Load the image from the binary data
image = Image.open(BytesIO(image_data))

# Set desired dimensions in inches
desired_width_in = 6  # Width in inches
desired_height_in = 8  # Height in inches
dpi = 300  # Desired DPI

# Calculate dimensions in pixels
desired_width_px = int(desired_width_in * dpi)
desired_height_px = int(desired_height_in * dpi)

# Resize the image to the desired dimensions
resized_image = image.resize((desired_width_px, desired_height_px), Image.LANCZOS)

# Save the resized image
resized_image.save("Fig2.png", dpi=(dpi, dpi))

# Optionally, display the resized image
resized_image.show()
