# `iaf_bw_2003` NEST extension module

Reference: Cavallari S, Panzeri S and Mazzoni A (2014) Comparison of the dynamics of neural interactions between current-based and conductance-based integrate-and-fire recurrent networks. Front. Neural Circuits 8:12. doi: 10.3389/fncir.2014.00012

This folder contains a local NEST extension module implementing
`iaf_bw_2003`, the conductance-based neuron model used in the cited paper
and by the `Cavallari_model` simulation example.

This folder follows a simplified NEST extension-module layout based on the NEST
example module repository: top-level `CMakeLists.txt` and `src/`.

## Build and install

```bash
cd examples/simulation/Cavallari_model/neuron_model
./install.sh
```

Internally this runs the documented flow:

```bash
mkdir -p build
cd build
cmake -Dwith-nest=$(command -v nest-config) ..
make
make install
```

This matches the current NEST extension-module example layout and load path:

```python
import nest
nest.Install("cavallari_module")
```
