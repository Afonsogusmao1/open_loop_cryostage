# 4.1 Control objective and front-position tracking

The trajectory-design problem was formulated as a model-based open-loop input-design problem. A finite-dimensional parameter vector \(\theta\) defines a requested cryostage reference-temperature trajectory, \(T_{\mathrm{ref}}(t)\), which is evaluated through the cryostage response model and the freezing solver. The trajectory is prescribed as a function of time and is not updated during a run using feedback from the freezing-front position. For each candidate \(\theta\), the computational cascade is

\[
\theta
\rightarrow T_{\mathrm{ref}}(t)
\rightarrow T_{\mathrm{plate}}(t)
\rightarrow T(r,z,t)
\rightarrow z_f(t)
\rightarrow J(\theta).
\]

Here, \(T_{\mathrm{plate}}(t)\) is the modelled plate-temperature trajectory imposed as the lower boundary condition in the freezing solver, \(T(r,z,t)\) is the simulated temperature field, \(z_f(t)\) is the post-processed centreline freezing-front position, and \(J(\theta)\) is the scalar objective minimized during trajectory design. The trajectory parameterization used to construct \(T_{\mathrm{ref}}(t)\) from \(\theta\) is described in Section 4.2.

Before solving the freezing problem, \(T_{\mathrm{ref}}(t)\) was converted into \(T_{\mathrm{plate}}(t)\) using the calibrated first-order plate-temperature response model obtained from the dry cryostage characterization assays described in Section 2.2.2. In those assays, the cryostage was driven to constant reference temperatures of \(-5\), \(-10\), \(-15\), and \(-20\,^\circ\mathrm{C}\), with three independent runs per setpoint. The fitted steady-state response \(T_{\mathrm{plate,ss}}(T_{\mathrm{ref}})\) represents the asymptotic plate temperature associated with each requested reference temperature. The fitted values of \(T_{\mathrm{plate,ss}}(T_{\mathrm{ref}})\) and \(\tau(T_{\mathrm{ref}})\) are reported in Table SX, and the corresponding dry-cooling traces with first-order model overlays are shown in Figure SX.

For a constant requested reference temperature, the model assumes that the plate temperature relaxes toward \(T_{\mathrm{plate,ss}}(T_{\mathrm{ref}})\) with setpoint-dependent response time \(\tau(T_{\mathrm{ref}})\), according to

\[
\frac{dT_{\mathrm{plate}}}{dt}
=
\frac{
T_{\mathrm{plate,ss}}(T_{\mathrm{ref}})-T_{\mathrm{plate}}
}{
\tau(T_{\mathrm{ref}})
}.
\]

In the sampled implementation used during trajectory design, this response was evaluated recursively as

\[
T_{\mathrm{plate},i}
=
\alpha_i T_{\mathrm{plate},i-1}
+
(1-\alpha_i)T_{\mathrm{plate,ss}}(T_{\mathrm{ref},i-1}),
\qquad
\alpha_i=\exp\left[-\frac{\Delta t_i}{\tau(T_{\mathrm{ref},i-1})}\right].
\]

where \(\Delta t_i=t_i-t_{i-1}\). Between the characterized reference temperatures, \(T_{\mathrm{plate,ss}}(T_{\mathrm{ref}})\) and \(\tau(T_{\mathrm{ref}})\) were obtained by linear interpolation of the fitted lookup values in Table SX. This model represents the experimentally observed plate-temperature response to a requested reference trajectory. It does not introduce feedback from the freezing front, and therefore the designed trajectory remains an open-loop prescribed thermal input to the model-based freezing workflow. This trajectory-design view is related to model-based Stefan-system tracking and inverse-solidification formulations in which boundary thermal histories are selected to obtain a desired interface motion [Pet22, Kan95].

The control objective was formulated in terms of front-position tracking rather than direct instantaneous front-velocity tracking. In the present workflow, the freezing front is obtained as a post-processed observable from the simulated temperature field, whereas velocity would require numerical differentiation of the extracted front-position trajectory. Because numerical differentiation of sampled data is sensitive to noise and often requires regularization, direct optimization of front velocity would be more sensitive to sampling interval, interpolation, thresholding, and discretization choices than tracking the front position itself [Bre20]. Front-velocity estimates were therefore treated as diagnostic quantities rather than primary optimization targets.

The target freezing behaviour was defined as a linear advance of the front through the filled sample height. This corresponds to an approximately constant average front velocity over the active interval without requiring velocity to be optimized directly. The reference front trajectory was defined as

\[
z_{f,\mathrm{ref}}(t)
=
H_{\mathrm{fill}}
\operatorname{clip}
\left(
\frac{t-t_0}{t_{\mathrm{end}}-t_0},
0,
1
\right),
\]

where \(H_{\mathrm{fill}}\) is the filled sample height, \(t_0\) is the start of the objective-evaluation window, and \(t_{\mathrm{end}}\) is the design horizon. This definition implies the average target velocity

\[
v_{\mathrm{ref}}
=
\frac{H_{\mathrm{fill}}}{t_{\mathrm{end}}-t_0}.
\]

The use of \(t_0\) allows the earliest part of the simulation to be excluded from the objective when needed, for example to avoid giving excessive weight to short initial transients immediately after filling.

For each candidate trajectory, the simulated front position was compared with the reference trajectory at the valid post-fill sample times. The main tracking term was

\[
J_{\mathrm{track}}(\theta)
=
\frac{1}{N_t}
\sum_{i=1}^{N_t}
\left[
\frac{
z_f(t_i;\theta)-z_{f,\mathrm{ref}}(t_i)
}{
H_{\mathrm{fill}}
}
\right]^2,
\]

where \(N_t\) is the number of time samples included in the objective calculation. Normalization by \(H_{\mathrm{fill}}\) makes the tracking error dimensionless and expresses deviations relative to the sample height. This choice is consistent with standard model-based tracking formulations, in which quadratic penalties are applied to deviations between predicted outputs and desired reference trajectories [Sch21].

The scalar objective minimized during optimization was then written as

\[
J(\theta)
=
w_{\mathrm{track}}J_{\mathrm{track}}(\theta)
+
w_{\mathrm{smooth}}J_{\mathrm{smooth}}(\theta)
+
w_{\mathrm{comp}}J_{\mathrm{comp}}(\theta),
\]

where \(w_{\mathrm{track}}\), \(w_{\mathrm{smooth}}\), and \(w_{\mathrm{comp}}\) are weighting factors. In the active linear full-process formulation, the tracking term is the primary objective term and the smoothness term acts as a secondary regularizer on the requested reference-temperature trajectory. Completion-related terms are implemented for alternative objective formulations and should be reported only when enabled in the optimization settings. Feasibility and admissibility constraints on the reference trajectory are described separately in Sections 4.2 and 4.3.

## References

[Pet22] B. Petrus, Z. Chen, H. El-Kebir, J. Bentsman, and B. Thomas, "Solid Boundary Output Feedback Control of the Stefan Problem: The Enthalpy Approach," IEEE Transactions on Automatic Control, vol. 68, pp. 3485-3500, Aug. 2022, doi: 10.1109/TAC.2022.3197704.

[Kan95] S. Kang and N. Zabaras, "Control of the freezing interface motion in two-dimensional solidification processes using the adjoint method," International Journal for Numerical Methods in Engineering, 1995, doi: 10.1002/NME.1620380105.

[Bre20] F. van Breugel, J. Kutz, and B. W. Brunton, "Numerical Differentiation of Noisy Data: A Unifying Multi-Objective Optimization Framework," IEEE Access, vol. 8, pp. 196865-196877, 2020, doi: 10.1109/ACCESS.2020.3034077.

[Sch21] M. Schwenzer, M. Ay, T. Bergs, and D. Abel, "Review on model predictive control: an engineering perspective," The International Journal of Advanced Manufacturing Technology, vol. 117, pp. 1327-1349, 2021, doi: 10.1007/s00170-021-07682-3.
