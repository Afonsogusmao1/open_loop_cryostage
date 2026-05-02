### State of the project

Afonso Gusmão has developed the numerical finite element simulation as well as the Bayesian optimization loop. I believe the infrastructure is mostly built, right now we just need to iterate and progressively solve the problem. Early results show that it is not so simple to solve that a three knot trajectory will work. We will need to be a bit more clever than that, which is good, because I was thinking the algorithm was too simple there was no innovation there besides the application.

### Next steps

What we need to focus now is on understanding the algorithm and learning to work with it. We need to figure a suite of results that we consistently look at to understand what is going and whether or not the optimization is achieving what we need it to. Here is a list of plots that would be useful to have during development and also for the paper:

* **A plot of the reference trajectory and the temperature of the plot over time.** This would allow us to see how well the cryostage is following the reference. If it is not following well, then we need to change the constraints or update the controller.
* **A plot of the freezing front vs time $z\_f(t)$, as well as its derivative, the front velocity $v\_f(t)$, both compared to the target velocity $v\_{ref}$.** These two are the key figures to have in the paper to demonstrate that our proposed solution works. Other summary statistics are important, but this is a figure every reader is expecting.
* **A plot of the tracking error $RMSE=\\sqrt{(v\_f(t)-v\_{ref})^2}$  (or another metric like relative error) over time for different target velocities.** Here we can show graphically how well the tracking works for different velocities in the same plot. It might be too messy or it might reveal a pattern in the errors. Color the lines with a colormap to easily distinguish lower from higher velocities (red to blue might be a good theme for the paper because of freezing).
* **A table or barplot or boxplot showing error vs target velocity.** A less messy version and we can show variation in the Bayesian optimization due to different seeds.

### Open questions

* The number of knots is still not solved, but I believe if we increase to $K=10$ it will work just fine.
* The scheduling of the knots is also not solved, but with a larger number of knots it will matter less. Another option is to search for the time instants as well, but it is hard to say which will lead to better results for the same number of variables. Either way, a larger number of knots will also make this alternative less impactful.
* We are currently optimizing with a rougher mesh and a larger time step. For the final results, we have three options:

  1. Simply accept the found temperature reference from the rougher grid and then run the simulation with the finer grid.
  2. Run a second optimization loop with the finer grid to optimize the temperature reference further.
  3. Adaptively refine the grid during the bayesian optimization.

  We can decide which to use by verifying how the mean tracking error changes as we refine the grid. Out of the three: option 1 will be fine if we no see the error does decrease significantly as we refine the grid; option 2 will likely lead to the best results; option 3 is the hardest to implement and I’m not sure it is applicable with Bayesian Optimization → Confer with Chat.

