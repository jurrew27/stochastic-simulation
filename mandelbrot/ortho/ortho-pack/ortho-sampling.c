#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include "mt19937.h"
#include "rand_support.h"

#define MAJOR	   9999
#define SAMPLES    (MAJOR * MAJOR)

/* We will have a total of SAMPLES samples, one in each of the MAJOR * MAJOR cells.
   Within each MAJOR row or column, we will have MAJOR minor rows/columns, such that
   each minor row/column contains only a single sample (it starts to resemble a
   sudoku puzzle...)
   So within each MAJOR row/column, we can use a permuation to allocate the points */

/* To keep it simple MAJOR is defined as a macro allowing the compiler to create 
   xlist and ylist, rather than having to do more complex things with calloc or
   malloc.
   This code assumes the use of the MT random generator, together with a few support
   functions in rand_support.c
*/
long xlist[MAJOR][MAJOR] = {{0}};
long ylist[MAJOR][MAJOR] = {{0}};

int
main()
{
    int i;
    int j;
    int k;
    int m;
    long double scale = 4.0 / ((long double) SAMPLES);
    double x;
    double y;

    init_genrand(3737);
    m = 0;
    /* Divide the entire area into MAJOR * MAJOR sub-squares */
    /* Divide each subsquare into MAJOR * MAJOR cells */
    /* The first index determines the row/column number of the subsquare */
    /* So xlist[2][5] would indicate what minor column contains the sample
       in the major cell with major column number 2 and major row number 5
       ylist[2][5] will provide its minor row number */

    for (i = 0; i < MAJOR; i++)
    {
        for (j = 0; j < MAJOR; j++)
        {
            xlist[i][j] = ylist[i][j] = m++;
        }
    }
    for (k = 0; k < RUNS; k++)
    {
        for (i = 0; i < MAJOR; i++)
        {
            /* permute will give me a permutation of the list with MAJOR elements */
            /* Due to the way the lists have been created subranges with similar
               x or y values will stay together
               xlist[i] is a pointer to the start of the 1D array xlist[i][0] ... xlist[i][MAJOR - 1]
             */
            permute(xlist[i], MAJOR);
            permute(ylist[i], MAJOR);
        }
        for (i = 0; i < MAJOR; i++)   /* Subsquare column */
        {
            for (j = 0; j < MAJOR; j++)  /* Subsquare row */
            {
                /* For a given subsquare column (i) every subsquare has its sample in
                   a different column of cells (as determined by xlist[i][j]
                   The random long double value selects a random point in the minor subsquare
                 */
                x = -2.0 + scale * (xlist[i][j] + (long double) genrand_real2());

                /* For a given subsquare row, every subsquare has its sample in a different 
                   row of cells */
                y = -2.0 + scale * (ylist[j][i] + (long double) genrand_real2());

                /* Do the desired computation with with x and y at this point in the code */
            }
        }
    }
    /* Postprocessing */
    return 0;
}
