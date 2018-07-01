SpatialWoE is a 2d Weight of Evidence transformation tool.
The goal is to significantly extend capabilities of simple linear models (like logreg) in a transparent and tractable way using variable transformation step.

One can use manual, quantiles or optimized (using decision trees) approach to transform 2 or more variables (visualization and manual corrections are ready only for 2d case now).

After transformation:
1) You can analyze the results of transformation using 2d heatmap (x,y axis are dedicated to variables, regions are highlighted with colors proportional to WoE).
2) Manually correct regions in 2d.

As the result, if the model transparency and tractability is mandatory (for example, credit scoring case), you can significantly extend flexibility of the model using SpatialWoE transformation.

See examples at the end of the Python code.

BR,
Denis Surzhko

