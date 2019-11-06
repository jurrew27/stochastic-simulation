/* In this file we will define some functions that come in useful when
   working with random distributions.
   1) Functions to select a random value from a sub-interval
   2) A function to create a permuted list
*/

/* Set the range for situations where multiple drawings from the
   same range are needed */

unsigned long
set_rand_range(unsigned long r);

/* Return the range as set above */

unsigned long
get_rand_range();

/* Return a random value from the range as set above */

unsigned long
get_subrange_rand();

/* Return a random value from a given range. Use when the range
   is used only once */

unsigned long
getrand_inrange(unsigned long range);

/* Permute a list of N longs */

void
permute(long list[], unsigned long N);
