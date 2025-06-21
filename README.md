## Coupled degradation mechanism aging modelling using PyBaMM

In this simulation, I used PyBaMM to model lithium-ion battery degradation over repeated charge–discharge cycles. The base model is the **Doyle-Fuller-Newman (DFN)** model, which I extended with key degradation mechanisms such as SEI growth, lithium plating, particle cracking, stress-driven loss of active material (LAM), and a lumped thermal model. Below is a breakdown of the physical models and governing equations I included, along with how the script is structured and what it computes.

---

### 1. Core Models

I started with the standard DFN model because it provides a comprehensive electrochemical framework. It includes lithium transport in solids and electrolytes, electrode kinetics, and potential distributions.

On top of this, I added:

* **Solvent-diffusion limited SEI growth**
* **Partially reversible lithium plating**
* **Stress-induced particle cracking**
* **LAM driven by mechanical stress**
* **A lumped thermal model** to account for heating

These mechanisms collectively simulate realistic aging behavior in the cell.

---

### 2. Differential Equations Governing the Model

#### 2.1 Lithium Diffusion in Particles

$$
\frac{\partial c_s}{\partial t} = \frac{D_s}{r^2} \frac{\partial}{\partial r} \left( r^2 \frac{\partial c_s}{\partial r} \right)
$$

This equation models lithium diffusion inside spherical electrode particles. It’s essential for capturing how lithium intercalates and deintercalates from active material.

#### 2.2 Electrolyte Transport and Potential

For the electrolyte, I used:

$$
\varepsilon \frac{\partial c_e}{\partial t} = \nabla \cdot (D_e \nabla c_e) + \frac{1 - t_+}{F} j
$$

$$
\nabla \cdot (\kappa \nabla \phi_e) = -j
$$

These equations describe lithium-ion transport and ionic conduction in the electrolyte.

#### 2.3 Reaction Kinetics: Butler-Volmer Equation

$$
j = j_0 \left[ \exp\left(\frac{\alpha_a F \eta}{RT}\right) - \exp\left(-\frac{\alpha_c F \eta}{RT}\right) \right]
$$

This governs the interfacial reaction rate and is central to computing current flow in the electrodes.

#### 2.4 SEI Growth

$$
\frac{d \delta_{\text{SEI}}}{dt} = -\frac{j_{\text{SEI}} M_{\text{SEI}}}{\rho_{\text{SEI}} F}
$$

This captures the growth of the SEI layer on the negative electrode. It consumes lithium irreversibly and increases impedance.

#### 2.5 Lithium Plating

$$
j_{\text{plate}} = k_{\text{plate}} \exp\left(-\frac{\alpha_{\text{plate}} F \eta_{\text{plate}}}{RT}\right)
$$

This equation represents lithium metal plating on the anode, which is partially reversible.

#### 2.6 Stress and Particle Cracking

$$
\sigma = E \epsilon + \frac{E \Omega c_s}{3(1-\nu)}
$$

This models stress in electrode particles due to swelling. Over time, it leads to mechanical failure and LAM.

#### 2.7 Loss of Active Material

$$
\frac{d \epsilon_{\text{AM}}}{dt} = -k_{\text{LAM}} |\sigma|
$$

Here, stress causes gradual degradation of electrode material, reducing capacity.

#### 2.8 Thermal Model

$$
\rho c_p \frac{dT}{dt} = Q - h A (T - T_{\text{amb}})
$$

This equation tracks the lumped temperature in the cell, based on heat generation and external cooling.

---

### 3. What the Code Does

I started by initializing the DFN model and enabling all relevant degradation options. I used the `OKane2022` parameter set (Parameter set that covers all aging mechanisms given above) and tweaked aging-related constants like SEI growth rate, LAM kinetics, and lithium plating rate to simulate realistic degradation.

I then defined a repeating experiment that included discharge at 1C, multi-stage charging (3C to 0.5C in steps), rest periods, and voltage limits. I ran this for five cycles (for demonstration, though it’s scalable to many more).

For solving the model, I used `CasadiSolver` with `atol=1e-2` and `rtol=1e-2`. This choice gave me a stable and fast solution for the stiff system of PDEs and ODEs.

Once the simulation was complete, I extracted data for:

* Cell voltage, current, capacity over time
* SEI thickness and lithium loss
* Plated lithium capacity (partially reversible)
* Active material volume fraction changes (LAM)
* Surface tangential stress in particles
* Lumped cell temperature

---

### 4. Cycle-Wise Aging Analysis

To quantify degradation, I analyzed each cycle’s discharge capacity, plated lithium, and LAM. I used this to calculate capacity fade (as a percent loss from initial capacity). The results were saved to a CSV file for post-processing or visualization.

---

### 5. Visualization and Final Stats

The script generated several plots:

* Voltage, current, and capacity vs. time
* Plated lithium and LAM vs. time
* Stress and temperature vs. time

Finally, I printed summary statistics including:

* Initial and final capacity
* Capacity fade
* Final LAM
* Max temperature
* Max stress

---

### 6. How the Model is Solved Internally

PyBaMM handles spatial discretization using finite volume methods. It creates an expression tree that links variables across space and time, and then translates the whole system into a set of algebraic equations. `CasadiSolver` solves these using an implicit method suited for stiff problems, with adaptive time-stepping.

---

## Results
The detailed results are given in [here](Results.md)



