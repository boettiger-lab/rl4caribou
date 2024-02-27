**Equations**

Moose:
$$dM/dt = M (r_m - \frac{(r_m \alpha_{mm} M)}{K_m} - \frac{(M^{x - 1} W a_M)}{(1 + B^x h_B a_B + M^x h_M a_M)} - \frac{(r_m \alpha_{mb} B)}{K_m} - \mu)$$

Caribou:
$$dB/dt = B(r_b- \frac{r_b \alpha_{bb} B}{K_b}- \frac{B^{x - 1} W a_B}{1 + M^x h_M a_M + B^x h_B a_B} - \frac{r_b \alpha_{bm}  M}{K_b})$$

Wolves:
$$dW/dt = W(\frac{B^x a_B}{1 + M^x h_M aM + B^x h_B a_B} + \frac{u a_M M^x)}{1 + M^x h_M aM + B^x h_B a_B} - d - \omega)$$

**Parameter Definitions**

+ W: Wolves
+ M: Moose 
+ B: Caribou
+ rb or rm: per capita intrinsic growth rate for caribou and moose, respectively units - #/indiv
+ $$\alpha_{ik}$$: the per capita impact of interaction of species k on species i  
+ $$\alpha_{ii}$$: the per capita impact of intraspecific competition (equal to one)
+ u: conversion factor - moose or primary prey might be bigger or more preferable than endangered species unitless?
+ x: type of functional response (1 = linear, 2 = logarithmic, 3 = logistic)  unitless
hM.
+ hB: PER CAPITA handling time of prey (moose, caribou) units - time/1 caribou|moose
+ $$\omega | \mu$$: per capita or percentage cull (wolves, moose) units - proportion
+ a :  predation efficiency = search efficiency * attack efficiency = km^2/time * (# successful kills)/(total # attempts) = units  km^2/time

**Notes**

+ alpha_bb ->1 & alpha_mm->1 here because we'll want to keep the meaning of Kb and Km as the carrying capacities in the absence of predation or competition.

+ We changed the structure of predation in the equation to reflect the overall time spent for predation depends both moose and caribou, thus if a wolf is currently eating a moose,  it cannot also be eating a caribou. This change was made after Niki had suspicions about the immediate lack of of impact moose culls on caribou. Niki thought that moose culls, while eventually good for caribou, might immediately be bad as the predation pressure on caribou would increase before wolves had a change to re-equilibrate to a decrease in prey (more predators than prey present).

