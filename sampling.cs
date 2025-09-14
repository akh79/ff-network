using System;
using System.Collections.Generic;

namespace ff_neural_net
{
    internal class Sampling
    {
        internal static int[] RandomKSubset(int n, int k, Random rand)
        {
            // Implements Floyd algorithm for k-subsets of n-set

            int[] result = new int[k];

            var selected = new HashSet<int>(k);

            for (int i = n - k, l = 0; i < n; ++i, ++l)
            {
                int t = rand.Next(0, i);
                if (selected.Contains(t))
                {
                    selected.Add(i);
                    result[l] = i;
                }
                else
                {
                    selected.Add(t);
                    result[l] = t;
                }
            }

            return result;
        }
    }
}
