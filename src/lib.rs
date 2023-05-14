//! A generalized Implementation of a Cuckoo Map.
//! # Cuckoo Maps
//! Cuckoo Maps are hash tables in which new entries (here called eggs) displace old colliding entries
//! (eggs) to a secondary hash table (here called branches) with its own hash function and all lookups
//! have a guaranteed complexity of O(1).
//!
//! #### In practice
//! - There can be more than two tables (branches)
//! - Eggs can be queued into buckets (here called nests)
//! - Neighboring buckets (nests) can be grouped for cache efficiency
//! - Instead of hashing multiple times, using (and reusing) different parts of the same hash is good enough
//! - Cuckoo Maps are very easy to grow by adding more branches
//! - Load factors above 99% are easy and practical to achieve
//! - It can act like parallel probabilistic LRU caches by e.g. decreasing the retainability of eggs
//!   which are being displaced from the last branch.
//! ##### Downsides
//! - Branch transitions may cause frequent cache misses
//! - Inserting an element into a CuckooMap, especially a full one, may be a very costly operation
//! - Cuckoo Maps have quite a lot of computational overhead
//! - Where their usage is possible, perfect hash function generators eat other designs for breakfast
//!
//! Overall this is my first programming project, so quite a lot of the downsides are influenced by
//! how little experience I have and the low code quality is likely the reason for the lack of
//! performance and may improve over time as I'm learning and seeking help and experienced opinions.

use std::hash::{BuildHasher, Hash};
use whyhash::{Finish2, HasherExt, WhyHash};

/// The datastructure itself and the necessary metadata
#[derive(Clone)]
pub struct CuckooMap<K, V, S>
where
    K: Hash,
    V: Default + Clone,
    S: BuildHasher,
{
    // The BuildHasher instance for the Cuckoo Map instance
    buildhasher: S,
    // The number of entries contained in the datastructure at any point in time
    len: usize,
    // How many eggs are queued in each nest
    eggcap_u8: u8,
    // When 2 is taken to the power of this number, it is the branch size
    log2_nestcap_u8: u8,
    // How many additional neighboring nests need to be considered on each branch for each hash
    nest_grouping_u8: u8,
    // How many branches there are in the Cuckoo Map
    branchcap_u8: u8,
    // A clock-like pointer pointing to the egg in each nest which should be displaced next
    in_nest_ptrs: Vec<u8>,
    // Holds a non-zero u8 hash for each egg.
    // Should make the Cuckoo Map A LOT more cache efficient if used properly
    meta: Vec<u8>,
    // The actual data (eggs)
    data: Vec<Entry<V>>,
    // Accept any hashable type as key
    k: std::marker::PhantomData<K>,
}

// A struct that uniquely identifies an egg by indicies, one for the Vec in_nest_ptrs
// and the other for the Vecs meta and data
#[doc(hidden)]
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Loc {
    ptr_loc: usize,
    data_loc: usize,
}

// A struct pointing to an egg's location
// (the entry itself and also the associated in_nest_ptr and meta-entry)
#[allow(dead_code)]
struct EntryLocRef<'a, V: Default + Clone> {
    in_nest_ptr: &'a mut u8,
    meta_entry: &'a mut u8,
    data_entry: &'a mut Entry<V>,
}

// It is intended to provide an Entry API implementation for this CuckooMap implementation.
// This conflicts with the "egg terminology" used since early versions, so at some point I
// intend to come up with better, less cringy terms, more inline with the rest of rust.

/// An Entry in the Cuckoo Map.
#[derive(Clone, Default)]
pub struct Entry<V: Default + Clone> {
    hash: u64,
    retainability: u8,
    value: V,
}

// An Enum which is used to describe what kind of transition was necessary to
// change from one nest-location to another
enum LocTrans<T> {
    InGroup(T),
    BetweenBranches(T),
    LastToFirstBranch(T),
}

/// The implementation of the datastructure
impl<K, V, S> CuckooMap<K, V, S>
where
    K: Hash,
    V: Default + Clone,
    S: BuildHasher,
{
    /// Returns an instance of a Cuckoo Map
    ///
    /// The eggcap are the amount of eggs in each nest. (In the furture)
    /// a value of 16 is recommended to optimize for SIMD accelerated lookups.
    ///
    /// Log2_nestcap, when used as exponent of 2, is the amount of nests on each branch.
    /// E.g. a value of 8 corresponds to a branchsize of 256 nests. A larger branchsize
    /// is more performant than additional branches, but new branches can be added quickly.
    /// Resizing is not yet supported (neither sizing nor adding branches).
    ///
    /// The nest_grouping parameter is how many additional neighboring nests an eggs can be
    /// located in. This enables even higher load-factors, especially for lower quality hash
    /// functions. On the other hand this option is probably only worth using if you need a
    /// small table. Both branchcap, but especially eggcap enable higher load factors aswell,
    /// give additional capacity on top and the latter is more performant, too.
    ///
    /// Branchcap is the amount of branches in the Cuckoo Map, each holding a number of nests.
    /// Branches enable scaling the table by a factor of the nestcap (times the eggcap).
    /// More branches also enable a higher load factor.
    ///
    /// Buildhasher becomes the table's buildhasher instance, such as RandomState or WhyHash.
    pub fn new_with_capacity_and_hasher(
        eggcap_u8: u8,
        log2_nestcap_u8: u8,
        nest_grouping_u8: u8,
        branchcap_u8: u8,
        buildhasher: S,
    ) -> CuckooMap<K, V, S> {
        let tmp1 = usize::pow(2, log2_nestcap_u8 as u32) * (branchcap_u8 as usize);
        let tmp2 =
            (eggcap_u8 as usize) * usize::pow(2, log2_nestcap_u8 as u32) * (branchcap_u8 as usize);
        let mut in_nest_ptrs = Vec::<u8>::with_capacity(tmp1);
        in_nest_ptrs.resize_with(tmp1, Default::default);
        let mut meta = Vec::<u8>::with_capacity(tmp2);
        meta.resize_with(tmp2, Default::default);
        let mut data = Vec::<Entry<V>>::with_capacity(tmp2);
        data.resize_with(tmp2, Default::default);
        CuckooMap {
            buildhasher,
            len: 0,
            eggcap_u8,
            log2_nestcap_u8,
            nest_grouping_u8,
            branchcap_u8,
            in_nest_ptrs,
            meta,
            data,
            k: std::marker::PhantomData,
        }
    }

    /// Returns the exact capacity of the CuckooMap.
    pub fn capacity(&self) -> usize {
        (self.eggcap_u8 as usize)
            * usize::pow(2, self.log2_nestcap_u8 as u32)
            * (self.branchcap_u8 as usize)
    }

    /// Returns the number of eggs currently in the CuckooMap
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if there are no (longer any) entries in the CuckooMap
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the load-factor of the CuckooMap
    pub fn load_factor(&self) -> f64 {
        (self.capacity() as f64) / (self.len() as f64)
    }

    /// Clears all entries from the CuckooMap.
    // I don't think it'll keep currently allocated memory, although I'd like it to in the future (safely?)
    pub fn clear(&mut self) {
        // Calculate the required capacities
        let tmp1 = usize::pow(2, self.log2_nestcap_u8 as u32) * (self.branchcap_u8 as usize);
        let tmp2 = (self.eggcap_u8 as usize)
            * usize::pow(2, self.log2_nestcap_u8 as u32)
            * (self.branchcap_u8 as usize);
        // Overwrite the previous vectors with the new ones
        // and also resize them to the required capacity
        self.in_nest_ptrs = Vec::<u8>::with_capacity(tmp1);
        self.in_nest_ptrs.resize_with(tmp1, Default::default);
        self.meta = Vec::<u8>::with_capacity(tmp2);
        self.meta.resize_with(tmp2, Default::default);
        self.data = Vec::<Entry<V>>::with_capacity(tmp2);
        self.data.resize_with(tmp2, Default::default);
        // Reset the number of entries in the CuckooMap to 0
        self.len = 0;
    }

    // A previously used, too slow function locating all nests for a hash.
    // It is retained for its simplicity.
    #[doc(hidden)]
    #[inline]
    pub fn locate_nests(&self, mut hash: u64) -> Vec<Loc> {
        // Calculate a mask large enough to address each nest on each branch
        let mask = u64::pow(2, self.log2_nestcap_u8 as u32) - 1;
        // Calculate the branchsize in number of entries for both the in_nest_ptrs Vec
        // and also the meta + data Vecs
        let bs_ptrs = usize::pow(2, self.log2_nestcap_u8 as u32);
        let bs_data = usize::pow(2, self.log2_nestcap_u8 as u32) * (self.eggcap_u8 as usize);
        // Allocate the Vector holding the indicies pointing to each of the hash's nests
        let mut nest_locs =
            Vec::<Loc>::with_capacity(self.branchcap_u8 as usize * self.nest_grouping_u8 as usize);
        // Locate one grouping of nests for each of the branches
        for i in 0..self.branchcap_u8 {
            // Get the hash's location on the current branch by masking the hash
            // then rotate the hash to get a "different hash" for the next branch
            let loc = (hash & mask) as usize;
            hash = hash.rotate_right(self.log2_nestcap_u8 as u32);
            // Locate all the nests in each branch's grouping
            // (+1 because 0 should disable just the "looking at neighboring nests feature", not all nests)
            for n in 0..self.nest_grouping_u8 + 1 {
                // Add the nest location on each branch to that branch's offset in the Vec.
                // Then add the nest's offset in its grouping
                // and wrap around to the beginning of the branch if the last nest was located at its end.
                nest_locs.push(Loc {
                    ptr_loc: (i as usize) * bs_ptrs + ((loc + (n as usize)) % bs_ptrs),
                    data_loc: (i as usize) * bs_data
                        + (((loc + (n as usize)) * (self.eggcap_u8 as usize)) % bs_data),
                });
            }
        }
        nest_locs
    }

    /// Return this CuckooMap instance's BuildHasher instance
    #[inline]
    pub fn hasher(&self) -> &S {
        &self.buildhasher
    }

    /// Hash one key with a Hasher generated by this CuckooMap instance's BuildHasher instance
    ///
    /// The returned Struct holds two hashes, one normal u64 and one u64,
    /// which is non-zero when cast to an u8
    #[inline]
    pub fn hash_key(&self, key: K) -> Finish2 {
        let mut hasher = self.buildhasher.build_hasher();
        key.hash(&mut hasher);
        hasher.finish2()
    }

    // A previous implementation of _get_mut()
    // It's no longer used because it's slower but retained because it's simple
    fn _get_mut_oldslow(
        &mut self,
        key: K,
        inestloc_onfail: usize,
    ) -> Result<EntryLocRef<V>, (usize, Finish2)> {
        // Hash the key and locate all nests across all branches
        let hash = self.hash_key(key);
        let nest_locs = self.locate_nests(hash.hash);

        // Iterate over the nests the specified key could be located in
        // (all nests in its group on every branch, wrapping at the branch's end)
        for loc in nest_locs.clone() {
            // Iterate over all entries (eggs) in each nest
            // Iteration could be stopped early if a 0 in the meta vector is found (empty spot),
            // because all entries are expected to occupy the first empty spot they encounter.
            // In practice this requires distinguishing deleted entries from empty ones,
            // so stopping early is not supported. Since these tables are expected to be only filled
            // and never deleted from (outdated entries being overwritten) and load factors of 100%
            // being very practical, there might be little use for not just iterating over everything
            for i in 0..(self.eggcap_u8 as usize) {
                // Check if the u8 in the meta table matches, only then check the actual entry
                if self.meta[loc.data_loc + i] == (hash.as_u8_nonzero as u8)
                    && self.data[loc.data_loc + i].hash == hash.hash
                {
                    // If both the meta hash and entry hash match, return the entry
                    // If the actual key matches (or the expected value) is ignored
                    // Such collisions are expected to occur extremely rarely and are
                    // assumed to be handled by the upstream application, because after
                    // such an insertion, the previous value would be returned instead
                    // of being written to the table as well.
                    return Ok(EntryLocRef {
                        in_nest_ptr: &mut self.in_nest_ptrs[loc.ptr_loc],
                        meta_entry: &mut self.meta[loc.data_loc + i],
                        data_entry: &mut self.data[loc.data_loc + i],
                    });
                }
            }
        }

        // If no match was found, return "the index of the nest" in the 'in_nest_ptr vector'
        // at the specified index in a "vector holding the 'indexes of all grouped nests across
        // all branches' the requested entry could be located in".
        // Also return the hash (struct Finish2) to avoid having to compute it twice
        Err((nest_locs[inestloc_onfail].ptr_loc, hash))
    }

    // Provides idential functionality to _get_mut_oldslow
    // It returns Ok(References to the Entry and (nest-)metadata) if an Entry with matching hash was found,
    // it returns an Err((index of nest in in_nest_ptrs, the new entry's hashes)) if there was no match
    // in case of an Err(), the returned in_nest_ptrs index is determined from the inestloc_onfail parameter,
    // which is the the index in a Vec holding all possible locations of a hash's nests.
    #[inline]
    fn _get_mut(
        &mut self,
        key: K,
        inestloc_onfail: usize,
    ) -> Result<EntryLocRef<V>, (usize, Finish2)> {
        // Hash the key. Two hashes are returned: One u64 and one u64 which is non-zero when cast to an u8
        let hash = self.hash_key(key);

        // Generate the mask for getting the nests loaction on each branch, then determine this location
        let mask = u64::pow(2, self.log2_nestcap_u8 as u32) - 1;
        let mut pos: usize = (hash.hash & mask) as usize;
        let mut nestloc_onfail: usize = 0;

        // Loop for the number of possible nest locations
        for i in 0..(self.branchcap_u8 as usize * (self.nest_grouping_u8 as usize + 1)) {
            // If the current iteration of the loop looks at the requested failure-nest, remember its position
            if i == inestloc_onfail {
                nestloc_onfail = pos;
            }

            // Determine the next nest's location from the current location and the hash
            // Match for all transition cases, since they are irrelevant here
            match self._next_innestptr_index(pos, hash.hash) {
                LocTrans::InGroup(x)
                | LocTrans::BetweenBranches(x)
                | LocTrans::LastToFirstBranch(x) => {
                    // Iterate over all eggs in the nest
                    for n in 0..(self.eggcap_u8 as usize) {
                        // Calculate the eggs position
                        let loc = pos * self.eggcap_u8 as usize + n;

                        // First look at nest's u8 hashes, there is huge (future) potential for SIMD here
                        // and only if there is a match look at the hash in the data Vec itself
                        if self.meta[loc] == (hash.as_u8_nonzero as u8)
                            && self.data[loc].hash == hash.hash
                        {
                            // If the hashes match, return references to the entry and associated data.
                            // Collisions are assumed to be handled by the application on insert!
                            return Ok(EntryLocRef {
                                in_nest_ptr: &mut self.in_nest_ptrs[pos],
                                meta_entry: &mut self.meta[loc],
                                data_entry: &mut self.data[loc],
                            });
                        }
                    }
                    // After all hashes have been checked for matches and if the function hasn't returned yet,
                    // proceed to the next nest
                    pos = x;
                }
            }
        }
        // Return the in_nest_ptrs index for the requested nest and the provided key's hashes
        Err((nestloc_onfail, hash))
    }

    // This function takes a hash and the index of a nest (of in_nest_ptrs) as arguments, assumes that
    // the hash was displaced from the given nest and calculates from this information the index of the
    // next nest for the given hash to be inserted at and the type of transition necessary to get there.
    //
    // This function can be called multiple times for the same hash to determine all nests locations for
    // the given hash, or it can be called for a different (displaced egg's) hash on each iteration,
    // calculating in which nest to insert a displaced egg next (and possibly displacing the next egg).
    #[inline]
    fn _next_innestptr_index(&self, lastpos: usize, hash: u64) -> LocTrans<usize> {
        // Calculate a mask large enough to address each nest on each branch
        let mask = u64::pow(2, self.log2_nestcap_u8 as u32) - 1;

        // Calculate the branch size (number of nests in in_nest_ptrs for each branch)
        let bs: usize = usize::pow(2, self.log2_nestcap_u8 as u32);

        // Calculate the quotient (which branch the nest was located on)
        // and the remainder (which location on the given branch the nest did occupy)
        let mut qlp: usize = lastpos / bs;
        let rem_lp: usize = lastpos % bs;

        // Calculate the position of the first nest in the grouping on each branch (hp = HashPosition)
        // The quotient is multiplied with the masksize, because the hash is roated once for each branch
        let hp = (hash.rotate_right(qlp as u32 * self.log2_nestcap_u8 as u32) & mask) as usize;

        // Determine the nest's position in each grouping (lg = LastGrouping)
        // Groupings can wrap around to the beginning of a branch
        let lg = if rem_lp < hp {
            // Add the remainder to the branchsize first, to avoid saturating at usize::MIN
            (rem_lp + bs) - hp
        } else {
            rem_lp - hp
        };

        // Determine if the next transition between nests is only within a group or between branches
        if lg < self.nest_grouping_u8 as usize {
            // Transition within a group by simply adding 1 to the last position, wrapping to the
            // beginning of the branch, if necessary. qlp * bs is the offset of the current branch.
            LocTrans::InGroup(qlp * bs + ((rem_lp + 1) % bs))
        } else {
            // Determine if a transition between branches needs to wrap from the last to the first branch
            qlp = (qlp + 1) % self.branchcap_u8 as usize;
            if qlp == 0 {
                // If the next nest is located on the first branch, its position is simply the masked hash
                LocTrans::LastToFirstBranch((hash & mask) as usize)
            } else {
                LocTrans::BetweenBranches(
                    qlp * bs
                        + (hash.rotate_right(qlp as u32 * self.log2_nestcap_u8 as u32) & mask)
                            as usize,
                )
            }
        }
    }

    // The purpose of this simple function is to insert an egg into the given nest and returning
    // the displaced egg, it's meta hash and at which location to insert it next (if wanted).
    //
    // If an egg is inserted into an empty position, this function simply returns "None"
    #[inline]
    fn _insert_next(
        &mut self,
        innestptr_index: usize,
        meta_entry: u8,
        entry: Entry<V>,
    ) -> Option<(LocTrans<usize>, u8, Entry<V>)> {
        // Calculate where into the given nest to insert the given egg. in_nest_ptrs
        // holds the index within each nest pointing at the next egg to be displaced
        let data_index =
            innestptr_index * self.eggcap_u8 as usize + self.in_nest_ptrs[innestptr_index] as usize;

        // Check if the position at which the egg is inserted is empty
        if self.meta[data_index] != 0 {
            // If the position is not empty, displace its egg by cloning it and its associated data,
            let tmp_entry: Entry<V> = self.data[data_index].clone();
            let tmp_meta_entry: u8 = self.meta[data_index];

            // calculating the next position to insert it at
            let nextpos: LocTrans<usize> =
                self._next_innestptr_index(innestptr_index, tmp_entry.hash);

            // and then writing the new egg at its position
            self.data[data_index] = entry;
            self.meta[data_index] = meta_entry;

            // Afterwards increment the in_nest_ptr for the nest by 1, wrapping at the nest border.
            // This way the in_nest_ptr always points either to the next empty position,
            // or to the egg having been in the nest the longest, making a nest a simple queue
            self.in_nest_ptrs[innestptr_index] =
                (self.in_nest_ptrs[innestptr_index] + 1) % self.eggcap_u8;

            // Return all information necessary to insert the displaced egg at its next nest
            Some((nextpos, tmp_meta_entry, tmp_entry))
        } else {
            // If there was no egg at the position indicated by in_nest_ptr, simply insert the egg
            // and increment in_nest_ptr by one, wrapping at the nest border
            self.data[data_index] = entry;
            self.meta[data_index] = meta_entry;
            self.in_nest_ptrs[innestptr_index] =
                (self.in_nest_ptrs[innestptr_index] + 1) % self.eggcap_u8;

            // By returning "None", the calling function knows that the egg was inserted
            // without displacing another one.
            None
        }
    }

    /// Check if an egg with the given key is in the CuckooMap
    ///
    /// This function can falsely return "true" if there is a hash collision.
    /// If collisions are likely (solve the Birthday Problem for u64::MAX; e.g. 0,01% for 60M eggs),
    /// it is recommended to use get() instead and handle collisions on insertionby tagging colliding
    /// entries (e.g. a collision counter which is then appended to the key before hashing)
    #[inline]
    pub fn contains_key(&mut self, key: K) -> bool {
        self._get_mut(key, 0).is_ok()
    }

    /// Get a non-mutable pointer to the value of the egg for the given key
    ///
    /// This function can falsely return a non-matching entry if there is a hash collision.
    /// Check contains_key() for details.
    #[inline]
    pub fn get(&mut self, key: K) -> Option<&V> {
        match self._get_mut(key, 0) {
            Err(_) => None,
            Ok(x) => Some(&x.data_entry.value),
        }
    }

    /// Get a tuple of non-mutable references to the hashed key, the value and the retainability of an egg.
    ///
    /// Whenever an egg is displaced from the last to the first branch, its retainability is decreased
    /// by one. When an egg with a retainability of 0 is displaced from the last branch during an
    /// insert(), insert() ends the insertion-chain and returns the displaced egg.
    ///
    /// This function can falsely return a non-matching entry if there is a hash collision.
    /// Check contains_key() for details.
    #[inline]
    pub fn get_key_value_retainability(&mut self, key: K) -> Option<(&u64, &V, &u8)> {
        match self._get_mut(key, 0) {
            Err(_) => None,
            Ok(x) => Some((
                &x.data_entry.hash,
                &x.data_entry.value,
                &x.data_entry.retainability,
            )),
        }
    }

    /// The mutable version of get(). Check get() for details.
    ///
    /// This function can falsely return a non-matching entry if there is a hash collision.
    /// Check contains_key() for details.
    #[inline]
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        match self._get_mut(key, 0) {
            Err(_) => None,
            Ok(x) => Some(&mut x.data_entry.value),
        }
    }

    /// The mutable version of get_key_value_retainability(). Check get_key_value_retainability() for details.
    ///
    /// This function can falsely return a non-matching entry if there is a hash collision.
    /// Check contains_key() for details.
    #[inline]
    pub fn get_mut_key_value_retainability(
        &mut self,
        key: K,
    ) -> Option<(&mut u64, &mut V, &mut u8)> {
        match self._get_mut(key, 0) {
            Err(_) => None,
            Ok(x) => Some((
                &mut x.data_entry.hash,
                &mut x.data_entry.value,
                &mut x.data_entry.retainability,
            )),
        }
    }

    /// Insert an egg into the CuckooMap with a retainability of 2.
    ///
    /// First, this triggers a search for an egg with the given key (or colliding hash).
    /// If another egg with the same hash is found, its value is replaced and the old value is
    /// returned as Ok(Some(old_value)). If collisions are likely (check contains_key() for details),
    /// it is recommended to handle them at this stage, e.g. by having a collision counter at the
    /// original entry and appending that counter to the displaced egg's key.
    /// Dedicated functions for handling collisions may be provided in the future.
    ///
    /// If no matching egg is found, an insertion-chain is started, which can be very costly if the
    /// load-factor approaches 100%, the table is large and average retainability is high. During the
    /// insertion-chain, eggs which are displaced from the last to the first branch have their
    /// retainability decreased by 1.
    ///
    /// If during the insertion-chain an empty spot is found Ok(None) is returned.
    ///
    /// If no empty spot is found the insertion-chain continues until an entry with a retainability of
    /// 0 is found. This is guaranteed to happen at some point, but may be costly. This entry's value
    /// is then returned as Err(value).
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>, V> {
        // Making the compiler happy
        let tmp_eggcap_u8 = self.eggcap_u8;
        // First try to find if an egg with the provided key already exists
        // If none exists, the index of the first nest (parameter "0") and the hashes are returned.
        match self._get_mut(key, 0) {
            Ok(x) => {
                // If a matching egg was found
                // Update its value, increment the in_nest_ptr wrapping at the nest border
                // and return the old value
                let tmp = x.data_entry.value.clone();
                x.data_entry.value = value;
                *x.in_nest_ptr = (*x.in_nest_ptr + 1) % tmp_eggcap_u8;
                Ok(Some(tmp))
            }
            Err((mut pos, hashed_key)) => {
                // Prepare all the necessary information to insert the egg
                // A retainability of 2 is chosen by default, this is somewhat arbitrary and may change later
                let mut meta = hashed_key.as_u8_nonzero as u8;
                let mut entry = Entry {
                    hash: hashed_key.hash,
                    retainability: 2,
                    value,
                };
                let mut ret: Result<Option<V>, V> = Ok(None);

                // Start of the insertion-chain. Sadly this might be very costly.
                //
                // I have thought about pre-computing a limited-depth insertion-chain for all branches
                // instead of just inserting on the first, in order to find the optimal insertion position,
                // but this gets very complicated when you account for the queue-like design of the nests
                // and some eggs possibly being displaced multiple times during the same chain.
                // Also there was no more guarantee of the insertion point and I'm unsure if it resulted in
                // fewer memory accesses on average than "just going for it".
                //
                // There was also the possibility of failing to insert after a fixed number of iterations,
                // but nest-groupings made it impossible to predict during which kind of transition between
                // nests the failure to insert would happen, but I'd like to guarantee a failure during the
                // transition from the last to the first branch, in order to avoid displacing too recent eggs.
                //
                // This leaves hybrid approaches, like triggering a failure condition after a fixed number of
                // iterations, but only interrupting the chain once a transition from the first to the last
                // branch happens. This way keeping a retainability value could be avoided, reducing complexity
                // and freeing a lot of memory for larger tables with small values. A downside is, that inserting
                // a new egg into an already full CuckooMap will always require as many iterations as allowed and
                // will always kick out the first egg displaced from the last branch without additional
                // considerations. This can be an upside to, since the amount of allowed iterations during an
                // insertion into a full CuckooMap can be reduced to 0.
                //
                // Another interesting approach would be keeping a running average of the retainability during
                // the insertion chain and returning as soon as an egg with a below average retainability
                // (possibly by a threshold (which might depend on the retainability values' variance)) is
                // displaced from the last branch. Depending on how retainability is gained, you'd want to add
                // a minimum threshold before an egg is permanently displaced from the CuckooMap to avoid
                // saturating retainability against u8::MAX.
                let mut b = true;
                while b {
                    // Insert the current egg at the given position and
                    // return the displaced egg and its next position if
                    // there is one
                    match self._insert_next(pos, meta, entry.clone()) {
                        None => {
                            // If an empty spot was found return Ok() and increment the CuckooMap's length
                            ret = Ok(None);
                            self.len += 1;
                            b = false;
                        }
                        Some((next_pos, next_meta, next_entry)) => {
                            // Update the loop-variables to their next value
                            meta = next_meta;
                            entry = next_entry;

                            // then check where the current egg was displaced from
                            match next_pos {
                                LocTrans::LastToFirstBranch(z) => {
                                    // If it was displaced from the last branch
                                    // check if its retainability has already reached 0.
                                    // If that's the case, return its value
                                    // otherwise decrease its retainability.
                                    if entry.retainability == 0 {
                                        ret = Err(entry.value.clone());
                                        b = false;
                                    } else {
                                        entry.retainability -= 1;
                                    }
                                    pos = z;
                                }
                                LocTrans::InGroup(z) | LocTrans::BetweenBranches(z) => pos = z,
                            }
                        }
                    }
                }
                ret
            }
        }
    }
}

/// The implementation of the datastructure for the case of the hasher being WhyHash
///
/// WhyHash is a very easy to understand hash function with a decent distribution and
/// security. It is ... *"derived"* from wyHash and various ... *"online resources"*.
impl<K, V> CuckooMap<K, V, WhyHash>
where
    K: Hash,
    V: Default + Clone,
{
    pub fn new_with_capacity(
        eggcap_u8: u8,
        log2_nestcap_u8: u8,
        nest_grouping_u8: u8,
        branchcap_u8: u8,
    ) -> CuckooMap<K, V, WhyHash> {
        Self::new_with_capacity_and_hasher(
            eggcap_u8,
            log2_nestcap_u8,
            nest_grouping_u8,
            branchcap_u8,
            WhyHash::new(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_some_data() {
        //CuckooMap

        let mut cm = CuckooMap::<usize, usize, WhyHash>::new_with_capacity(8, 16, 1, 3);
        println!("CM capacity: {}", cm.capacity());
        for i in 0..cm.capacity() {
            cm.insert(i, i);
        }
        println!("CM length:   {}", cm.len());
        let bench_cm = std::time::Instant::now();
        for i in 0..cm.capacity() {
            if !cm.contains_key(i) {
                panic!("CM: Key {} not found", i);
            }
        }
        let bench_cm_done = bench_cm.elapsed().as_secs_f64();

        //HashMap

        let mut hp = std::collections::HashMap::<usize, usize>::with_capacity(cm.capacity());

        for i in 0..cm.capacity() {
            hp.insert(i, i);
        }

        let bench_hm = std::time::Instant::now();
        for i in 0..cm.capacity() {
            if !hp.contains_key(&i) {
                panic!("HP: Key {} not found", i);
            }
        }
        let bench_hm_done = bench_hm.elapsed().as_secs_f64();

        //BTreeMap

        let mut bm = std::collections::BTreeMap::<usize, usize>::new();
        for i in 0..cm.capacity() {
            bm.insert(i, i);
        }

        let bench_bm = std::time::Instant::now();
        for i in 0..cm.capacity() {
            if !bm.contains_key(&i) {
                panic! {"BM: Key {} not found", i};
            }
        }
        let bench_bm_done = bench_bm.elapsed().as_secs_f64();

        println!(
            "M Lu/s CM: {}",
            cm.capacity() as f64 / bench_cm_done / 1_000_000.0
        );
        println!(
            "M Lu/s HM: {}",
            cm.capacity() as f64 / bench_hm_done / 1_000_000.0
        );
        println!(
            "M Lu/s BM: {}",
            cm.capacity() as f64 / bench_bm_done / 1_000_000.0
        );
        println!("CM/HM div: {}", bench_cm_done / bench_hm_done);
    }
}
