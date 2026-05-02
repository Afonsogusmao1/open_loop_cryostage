# Robust n8 Bayesian optimization with direct-speed objective alignment

## Motivation

The previous n8 Bayesian optimization (BO) campaigns showed a strong
seed-to-seed variability in the final selected temperature profiles. This
variability should not be interpreted as stochasticity of the physical model:
for a fixed cryostage reference trajectory, the finite-element simulation is
deterministic. Instead, the variability was traced to the optimization
formulation. Different random seeds changed the initial design and the
acquisition trajectory of the BO, and the scalar objective did not explicitly
penalize the metric used to judge success, namely the achieved direct freezing
front speed.

The revised formulation therefore keeps the same deterministic simulator, but
modifies the scalar objective and the search protocol so that the BO directly
optimizes the experimentally relevant quantity.

## Temperature control parameterization

Let

$$
\theta = (\theta_1,\theta_2,\ldots,\theta_N)
$$

denote the cryostage reference temperatures at fixed knot times

$$
0=t_1<t_2<\cdots<t_N=t_f .
$$

For the present n8 campaign, \(N=8\). The continuous reference trajectory is
defined by piecewise-linear interpolation,

$$
T_{\mathrm{ref}}(t) =
\mathrm{Interp}\left\{(t_i,\theta_i)\right\}_{i=1}^{N}.
$$

The trajectory is constrained to remain monotone non-increasing in time,

$$
\theta_1 \geq \theta_2 \geq \cdots \geq \theta_N,
$$

so that the reference temperature either remains constant or becomes colder as
the freezing process progresses.

The BO does not optimize the physical temperatures directly. Instead, it
samples a normalized vector

$$
u=(u_1,u_2,\ldots,u_N), \qquad u_i\in[0,1],
$$

which is mapped recursively into physical temperatures. If
\([L_i,U_i]\) are the node-wise temperature bounds, then the last knot is
computed as

$$
\theta_N = L_N + u_N (U_N-L_N),
$$

and for \(i=N-1,\ldots,1\),

$$
\theta_i =
\max(L_i,\theta_{i+1})
+ u_i\left[U_i-\max(L_i,\theta_{i+1})\right].
$$

This mapping guarantees, by construction, that every BO candidate satisfies the
temperature bounds and the monotonicity constraint before the expensive
simulation is evaluated.

## Direct-speed metric

Let \(z_f(t)\) be the simulated freezing front position. The control interval is
defined by two depths,

$$
z_{\min}=2.5~\mathrm{mm}, \qquad z_{\max}=11.5~\mathrm{mm}.
$$

The first crossing times are

$$
t_{\min} = \inf\{t: z_f(t)\geq z_{\min}\},
$$

and

$$
t_{\max} = \inf\{t: z_f(t)\geq z_{\max}\}.
$$

The achieved direct speed is then

$$
\hat v =
\frac{z_{\max}-z_{\min}}{t_{\max}-t_{\min}}.
$$

For a prescribed target speed \(v^\star\), the signed relative direct-speed
error is

$$
\delta_v = \frac{\hat v-v^\star}{v^\star},
$$

and the absolute relative error is

$$
e_v = |\delta_v|.
$$

## Previous objective and limitation

The previous scalar objective used by the BO was primarily a front-tracking
objective,

$$
J_{\mathrm{old}}(\theta)
=
w_z J_z(\theta)
+ w_c J_c(\theta)
+ w_s J_s(\theta),
$$

where \(J_z\) is the normalized front-tracking mean-square error, \(J_c\) is an
incomplete-freezing or completion penalty, and \(J_s\) is a smoothness penalty
on the reference temperature trajectory.

The front-tracking term was computed from the deviation between the simulated
front and a constant-speed reference front,

$$
J_z =
\frac{1}{|\mathcal T|}
\sum_{t\in\mathcal T}
\left[
\frac{z_f(t)-z_{\mathrm{ref}}(t)}
{z_{\max}-z_{\min}}
\right]^2 .
$$

The smoothness term was computed from the slopes between consecutive control
knots,

$$
J_s =
\frac{1}{N-1}
\sum_{i=1}^{N-1}
\left[
\left(
\frac{\theta_{i+1}-\theta_i}{t_{i+1}-t_i}
\right)
\left(
\frac{t_f}{T_{\max}-T_{\min}}
\right)
\right]^2 .
$$

This objective can select a profile with relatively good smoothness and
front-tracking behavior even if its achieved direct speed is not the closest to
the prescribed target. In the n8 high-speed probe at \(v^\star=0.013\)
mm/s, several seeds did contain evaluated candidates with direct-speed errors
below approximately 1-2%, but the old objective selected different candidates
because the direct-speed error was not explicitly represented in \(J\).

## Revised objective with direct-speed penalty

The revised objective adds an explicit penalty on the direct-speed error,

$$
J_{\mathrm{new}}(\theta)
=
J_{\mathrm{old}}(\theta)
+ w_v J_v(\theta).
$$

The speed penalty uses a deadband tolerance \(\tau_v\). Errors inside the
tolerance are not penalized, so that the optimizer can still prefer smoother or
better-tracking profiles once the speed is sufficiently accurate:

$$
J_v(\theta)
=
\left[
\max(e_v-\tau_v,0)
\right]^2 .
$$

If the simulated front does not reach both \(z_{\min}\) and \(z_{\max}\), then
the direct-speed penalty is replaced by the incomplete-run penalty. In the
robust campaign reported here,

$$
w_v=50,
\qquad
\tau_v=0.01,
$$

corresponding to a 1% direct-speed tolerance.

Therefore, outside the 1% deadband, the BO is forced to treat direct-speed
error as a primary component of the objective. Inside the deadband, smoothness
and front-tracking remain useful secondary criteria.

## Deterministic initial portfolio

To reduce seed-to-seed variability, the initial design is made deterministic.
For each target speed, a physically plausible target-dependent seed profile
\(\theta^{(0)}(v^\star)\) is chosen and mapped into normalized coordinates,

$$
u^{(0)} = \Phi^{-1}(\theta^{(0)}),
$$

where \(\Phi\) denotes the monotone unit-box mapping described above.

The initial portfolio is then generated by deterministic local perturbations
around \(u^{(0)}\),

$$
u_k^{\mathrm{init}}
=
\mathrm{clip}
\left(
u^{(0)}
+ \alpha_k s_k,
0,1
\right),
$$

where \(s_k\in\{-1,+1\}^{N}\) is a deterministic sign pattern and
\(\alpha_k\) is a scaled perturbation amplitude. In the present campaign,
20 deterministic initial points were used. This means that every seed starts
from the same deterministic local portfolio before the seed-dependent BO
acquisition loop begins.

This step is important because the initial design can strongly influence the
Gaussian-process surrogate and, consequently, the acquisition path.

## Bayesian optimization loop

After evaluating the seed profile and deterministic initial portfolio, BO is
performed in the normalized space \(u\in[0,1]^N\). The optimizer minimizes
\(J_{\mathrm{new}}(\Phi(u))\), equivalently maximizing its negative value.

Expected improvement (EI) was used as the acquisition function. In maximization
form, if \(Y(u)\) is the Gaussian-process surrogate for the optimizer target and
\(Y_{\mathrm{best}}\) is the best observed target value, then

$$
\mathrm{EI}(u)
=
\mathbb E
\left[
\max\left(Y(u)-Y_{\mathrm{best}}-\xi,0\right)
\right],
$$

with

$$
\xi=0.01.
$$

The next BO point is selected as

$$
u_{k+1}
=
\arg\max_{u\in[0,1]^N}
\mathrm{EI}(u).
$$

Each run used 60 acquisition-guided BO iterations.

## Local refinement stage

The final 20 evaluations are not additional global BO iterations. They are a
deterministic local refinement stage around the best candidate found so far.

Let

$$
u_{\mathrm{best}}
=
\arg\min_{u\in\mathcal D}
J_{\mathrm{new}}(\Phi(u)),
$$

where \(\mathcal D\) is the set of all candidates already evaluated by the seed
point, initial portfolio, and BO loop.

The refinement candidates are deterministic perturbations around
\(u_{\mathrm{best}}\),

$$
u_k^{\mathrm{ref}}
=
\mathrm{clip}
\left(
u_{\mathrm{best}}
+ \beta s_k,
0,1
\right),
$$

with a small local amplitude. In the present campaign,

$$
\beta = 0.04.
$$

Each refinement candidate is mapped back into the physical temperature profile
and evaluated with the same deterministic simulator. The final selected
solution is

$$
u^\star
=
\arg\min_{u\in\mathcal D\cup\mathcal R}
J_{\mathrm{new}}(\Phi(u)),
$$

where \(\mathcal R\) is the set of local refinement candidates.

The purpose of this final stage is to polish the best BO solution locally,
without restarting exploration over the full search space. It also makes the
final selection less sensitive to the last stochastic acquisition suggestions.

## Final campaign configuration

The robust n8 campaign used the following structure per run:

$$
1~\theta^{(0)}
+ 20~\mathrm{initial}
+ 60~\mathrm{BO}
+ 20~\mathrm{local~refinement}
= 101~\mathrm{evaluations}.
$$

The full campaign used seven target speeds and five mandatory seeds,

$$
v^\star \in
\{0.007,0.008,0.009,0.010,0.011,0.012,0.013\}
~\mathrm{mm/s},
$$

and

$$
\mathrm{seed}\in\{17,29,41,53,67\}.
$$

Therefore, the campaign contained

$$
7\times 5\times 101 = 3535
$$

simulation evaluations.

The characterization admissibility precheck was disabled for this diagnostic
campaign, while the monotone n8 parameterization and the global temperature
bounds \([-21,0]\,^\circ\mathrm{C}\) were retained.

## Robust-campaign results

The final n8 robust direct-speed campaign produced the following summary:

| Target speed (mm/s) | Median achieved (mm/s) | Min achieved | Max achieved | Seed spread | Max error (%) |
|---:|---:|---:|---:|---:|---:|
| 0.007 | 0.007057 | 0.006923 | 0.007077 | 0.000154 | 1.10 |
| 0.008 | 0.008050 | 0.007994 | 0.008062 | 0.000068 | 0.78 |
| 0.009 | 0.009095 | 0.009043 | 0.009106 | 0.000064 | 1.18 |
| 0.010 | 0.010038 | 0.009908 | 0.010146 | 0.000239 | 1.46 |
| 0.011 | 0.011083 | 0.010997 | 0.011349 | 0.000352 | 3.17 |
| 0.012 | 0.012044 | 0.012044 | 0.012044 | 0.000000 | 0.36 |
| 0.013 | 0.013003 | 0.013003 | 0.013003 | 0.000000 | 0.02 |

All targets were achieved within 5% across all five seeds. The high-speed
targets, which previously exhibited a plateau and strong seed dependence, were
substantially stabilized: for \(0.012\) and \(0.013\) mm/s, the five seeds
selected identical final speeds in the coarse optimization campaign.

## Interpretation

The key conclusion is that the earlier seed dependence was not evidence that
the physical model itself was seed-dependent. Rather, it indicated that the BO
formulation was underconstrained with respect to the quantity used for judging
success. By adding an explicit direct-speed penalty and reducing the
seed-dependence of the initial design, the optimization became much more
reproducible.

The revised formulation can be summarized as follows:

1. Use a deterministic simulator and monotone n8 temperature trajectories.
2. Optimize a scalar objective that explicitly includes achieved direct-speed
   error.
3. Use a deterministic initial portfolio around a target-dependent seed
   trajectory.
4. Use BO for global/local surrogate-guided search.
5. Finish with deterministic local refinement around the incumbent.

This provides a more robust basis for subsequent fine-resolution confirmations
and for comparing different knot schedules or admissibility constraints.
