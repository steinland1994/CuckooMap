A generalized Implementation of a Cuckoo Map.
# Cuckoo Maps
Cuckoo Maps are hash tables in which new entries (here called eggs) displace old colliding entries
(eggs) to a secondary hash table (here called branches) with its own hash function and all lookups
have a guaranteed complexity of O(1).

#### In practice
- There can be more than two tables (branches)
- Eggs can be queued into buckets (here called nests)
- Neighboring buckets (nests) can be grouped for cache efficiency
- Instead of hashing multiple times, using (and reusing) different parts of the same hash is good enough
- Cuckoo Maps are very easy to grow by adding more branches
- Load factors above 99% are easy and practical to achieve
- It can act like parallel probabilistic LRU caches by e.g. decreasing the retainability of eggs which are being displaced from the last branch.
##### Downsides
- Branch transitions may cause frequent cache misses
- Inserting an element into a CuckooMap, especially a full one, may be a very costly operation
- Cuckoo Maps have quite a lot of computational overhead
- Where their usage is possible, perfect hash function generators eat other designs for breakfast

Overall this is my first programming project, so quite a lot of the downsides are influenced by
how little experience I have and the low code quality is likely the reason for the lack of
performance and may improve over time as I'm learning and seeking help and experienced opinions.
