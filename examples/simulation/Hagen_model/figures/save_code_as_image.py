import subprocess
import os
from pdf2image import convert_from_path

# Python code to be included in the LaTeX document
code = """
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
"""

# Create a LaTeX document
latex_code = f"""
\\documentclass{{article}}
\\usepackage{{minted}}
\\usepackage{{geometry}}

% Set page dimensions to match the image size
\\geometry{{papersize={{3.5in,6.5in}}, margin=0.1in}}

\\begin{{document}}

\\begin{{minted}}[fontsize=\\footnotesize, breaklines]{{python}}
{code}
\\end{{minted}}

\\end{{document}}
"""

# Write the LaTeX code to a file
with open("code_document.tex", "w") as tex_file:
    tex_file.write(latex_code)

# Compile the LaTeX file to PDF
try:
    subprocess.run(
        ["pdflatex", "-shell-escape", "code_document.tex"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print("PDF generated successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error during PDF generation: {e.stderr.decode()}")

# Convert PDF to High-Resolution Image
pdf_file = "code_document.pdf"
output_image_prefix = "output_image"
try:
    # Convert PDF to images
    images = convert_from_path(pdf_file, dpi=300)
    for i, image in enumerate(images):
        image_path = f"{output_image_prefix}_page_{i + 1}.png"
        image.save(image_path, "PNG")
        print(f"Saved high-resolution image: {image_path}")
except Exception as e:
    print(f"Error during PDF-to-image conversion: {e}")

# Step 4: Cleanup intermediate files (optional)
for ext in [".aux", ".log", ".out", ".tex"]:
    try:
        os.remove(f"code_document{ext}")
    except FileNotFoundError:
        pass