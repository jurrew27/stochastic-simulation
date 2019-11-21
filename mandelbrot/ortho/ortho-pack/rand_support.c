/* In this file we will define some functions that come in useful when
   working with random distributions.
   1) Functions to select a random value from a sub-interval
   2) A function to create a permuted list
*/

#include <stdio.h>
#include "mt19937.h"
#include "rand_support.h"

static unsigned long range = MAX_UNSIGNED_RAND;
static unsigned long reject_above = MAX_UNSIGNED_RAND;
static unsigned long divisor = 1;

/* Use this function if you want to select a number of
   samples from the same range, to define the range.
   The random values will be uniformly distributed over
   0..r and may include both 0 and r */
unsigned long
set_rand_range(unsigned long r)
{
    unsigned long long N = MAX_UNSIGNED_RAND;
    unsigned long long rp1;
    N++;
    if (r)
    {
        rp1 = r;
        rp1++;
        divisor = N / rp1;
        reject_above = rp1 * divisor - 1;
        range = r;
    }
    return r;
}

/* If you forgot which range you set */
unsigned long
get_rand_range()
{
    return range;
}

/* Get your random value */
unsigned long
get_subrange_rand()
{
    unsigned long r;
    do
    {
        r = genrand_int32();
    } while (r > reject_above);
    return (r / divisor);
}

/* Use for a single sample from a given range, does not
   affect range settings.
   The random values will be uniformly distributed over
   0..range and may include both 0 and range */
unsigned long
getrand_inrange(unsigned long range)
{
    unsigned long long N = MAX_UNSIGNED_RAND;
    unsigned long long rp1;
    unsigned long divisor;
    unsigned long reject_above;
    unsigned long r;
    if (range)
    {
        N++;
        rp1 = range;
        rp1++;
        divisor = N / rp1;
        reject_above = rp1 * divisor - 1;
        do
        {
            r = genrand_int32();
        } while (r > reject_above);
        return (r / divisor);
    }
    return 0;
}

/* Permute a list of N longs */

void
permute(long list[], unsigned long N)
{
    unsigned long i;
    long h;
    unsigned long r;
    for (i = N-1; i; i--)
    {
        h = list[i];
        r = getrand_inrange(i);
        list[i] = list[r];
        list[r] = h;
    }
}

/*  testing code
int
main()
{
    unsigned long i;
    int j;
    long list[100];
    init_genrand(19376);
    for (i = 0; i < 25; i++)
    {
        printf("%lu ", getrand_inrange(10));
    }
    printf("\n");
    set_rand_range(12);
    for (i = 0; i < 25; i++)
    {
        printf("%lu ", get_subrange_rand());
    }
    printf("\n");
    for (i = 1; i < 22; i++)
    {
        for (j = 0; j < i+5; j++)
        {
            list[j] = j;
        }
        permute(list, i+5);
        for (j = 0; j < i+5; j++)
        {
            printf("%ld ", list[j]);
        }
        printf("\n");
    }
    return 0;
}
*/
