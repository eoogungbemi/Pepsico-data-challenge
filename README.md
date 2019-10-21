### Pepsico data challenge
 Pepsico data challenge
Challenge Statement

Solvers will create a model that predicts the shelf life of snack products. Products change as they age and ultimately taste different from the time they were packaged. There are many potential factors that may influence how a product ages: the base ingredient; how a product is processed; the stability of the ingredients used when making the product, etc.

Snack product shelf life is determined with a specific test protocol. As samples age, a sensory panel evaluates the aged sample versus a fresh sample. The column in the dataset named “Difference From Fresh” is the response from the sensory panel. Without going into the details of how this measure is generated, when this value is 20 or greater the aged sample at that time point is considered different enough from fresh that the product has reached its maximum shelf life.

The dataset available to create a predictive shelf life model is comprised of 81 individual shelf life studies (Identified by the Study Number column) across a wide variety of snack products. Each of these studies has one or more samples (identified by the Sample ID column) that were aged and evaluated by the sensory panel. The other columns in the data set are:

Product Type: A, B, C, etc. are different product formats (examples: cookie, cracker)
Base Ingredient: A, B, C, etc. are different base ingredients (examples: wheat, corn)
Process Type: A, B, C, etc. are different ways a product can be processed (examples: dried, fried)
Sample Age (in weeks)
Difference From Fresh (>20 indicates no longer fresh)
Storage Conditions: Cold Climate, Warm Climate, High Temperature and Humidity
Package Stabilizer Added: Y, N (some products are packaged with a stability agent)
Transparent Window in Package: Y, N (light permeability in a package may influence shelf life)
Processing Agent Stability Index: continuous measure of the stability of the processing agent used
Preservative Added: Y, N (some products have an added preservative)
Moisture (%): measured on aged sample
Residual Oxygen (%): measured on aged sample
Hexanal (ppm): measured on aged sample

As you will see there is a substantial amount of missing data, this information was either not measured or not captured.

A useful shelf life model would allow a product developer to predict shelf life based on the product, process, packaging information, and storage conditions related to where the product will be sold. A developer may choose to add a packaging stabilizer, add or remove a transparent window in the bag, use a processing agent with a different stability index and add a preservative to extend the shelf life or reduce production costs depending on how these factors impact the product shelf life.
