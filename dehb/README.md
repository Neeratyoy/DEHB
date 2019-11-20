DEHB versions:

- **Version 1**
    - Each DEHB iteration begins with a random population sampled for 
    the population size determined by Successive Halving (SH).
    - DE is run for _5 generations_ for each SH iteration inside each
    such DEHB iteration. 

- **Version 2**
    - Only the first DEHB iteration's first SH iteration is initialized
     with a random population.
    - In each of the SH iterations, the top _n%_ (as selected by SH 
    budget spacing) is passed on to the next budget.
    - DE is performed on this population for _1 generation_ and the
    single large population is updated with these evolved individuals.
    - The next SH iteration then selects the top _n%_ from the evolved
    individuals from the previous SH steps and so on.

- **Version 3**
    - The total number of budget/configuration spacing from SH is
    determined at the beginning and that many sets of population are
    maintained, i.e., each population set consists of individuals 
    evaluated on the same budget.
    - These population sets are created during the first DEHB iteration.
    - For any j-th SH iteration (j>1) in any i-th DEHB iteration (i>1),
    the top _n%_ individuals are forwarded to the j+1-th SH iteration.
    - These individuals are evaluated for the budget for j+1-th SH step
    and then put into a pool along with the current population from 
    (i-1)-th DEHB iteration for the corresponding budget, and ranked 
    where the top half is retained.
    - These individuals undergo DE for _1 generation_ and the top _n %_
    is forwarded to the next step and so on.  
