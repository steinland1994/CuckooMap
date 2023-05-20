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

#[doc(hidden)]
pub trait EntryTrait: Default + Clone {
    type V;
    fn hashes(&self) -> &Finish2;
    fn value(&mut self) -> &mut Self::V;
}

#[doc(hidden)]
pub trait BranchEntryTrait {
    type E: EntryTrait;
    fn new(entrycap: u8) -> Self;
    fn ptr_displace(&self) -> u8;
    fn is_full(&self) -> bool;
    fn clear(&mut self);
    fn remove(&mut self, hashes: &Finish2) -> Option<Self::E>;
    fn get(&mut self, hashes: &Finish2) -> Option<&mut Self::E>;
    fn insert(&mut self, entry: Self::E) -> Option<Self::E>;
}

#[doc(hidden)]
pub trait BranchTrait {
    type E: EntryTrait;
    type BE: BranchEntryTrait;
    fn new(bucket_entrycap: u8, log2_bucketcap: u8, bucket_grouping: u8, hash_rotator: u32)
        -> Self;
    fn capacity(&self) -> usize;
    fn length(&self) -> usize;
    fn clear(&mut self);
    fn remove(&mut self, hashes: &Finish2) -> Option<Self::E>;
    fn get(&mut self, hashes: &Finish2) -> Option<&mut Self::E>;
    fn insert(&mut self, entry: Self::E) -> Option<Self::E>;
}

#[doc(hidden)]
#[derive(Default, Clone)]
pub struct Entry<V>
where
    V: Default + Clone,
{
    hashes: Finish2,
    value: V,
}

#[doc(hidden)]
pub struct Bucket<V>
where
    V: Default + Clone,
{
    displace_index: u8,
    meta: Vec<u8>,
    data: Vec<Entry<V>>,
}

#[doc(hidden)]
pub struct BranchWithBucketGroups<V, B>
where
    V: Default + Clone,
    B: BranchEntryTrait,
{
    hash_rotator: u32,
    mask: u64,
    bucket_grouping: u8,
    bucket_entrycap: u8,
    cap_buckets: usize,
    length: usize,
    data: Vec<B>,
    value_type: std::marker::PhantomData<V>,
}

#[doc(hidden)]
pub struct BranchWithoutBucketGroups<V, B>
where
    V: Default + Clone,
    B: BranchEntryTrait,
{
    hash_rotator: u32,
    mask: u64,
    bucket_entrycap: u8,
    cap_buckets: usize,
    length: usize,
    data: Vec<B>,
    value_type: std::marker::PhantomData<V>,
}

pub struct CuckooMap<K, V>
where
    K: Hash,
    V: Default + Clone,
{
    key: std::marker::PhantomData<K>,
    value: std::marker::PhantomData<V>,
}

impl<K, V> CuckooMap<K, V>
where
    K: Hash,
    V: Default + Clone,
{
    pub fn new_with_buckets(
        num_branches: u8,
        log2_branch_size: u8,
        bucket_size: u8,
    ) -> PrvCuckooMap<K, V, WhyHash, BranchWithoutBucketGroups<V, Bucket<V>>> {
        PrvCuckooMap::<K, V, WhyHash, BranchWithoutBucketGroups<V, Bucket<V>>>::new(
            num_branches,
            log2_branch_size,
            bucket_size,
            0,
            WhyHash::new(),
        )
    }
}

#[doc(hidden)]
pub struct PrvCuckooMap<K, V, S, M>
where
    K: Hash,
    V: Default + Clone,
    S: BuildHasher,
    M: BranchTrait<E = Entry<V>>,
{
    //
    key: std::marker::PhantomData<K>,
    value: std::marker::PhantomData<V>,
    buildhasher: S,
    branches: Vec<M>,
}

// ############################################################################################################
impl<K, V, S, M> PrvCuckooMap<K, V, S, M>
where
    K: Hash,
    V: Default + Clone,
    S: BuildHasher,
    M: BranchTrait<E = Entry<V>>,
{
    fn new(
        num_branches: u8,
        log2_branch_size: u8,
        bucket_size: u8,
        bucket_grouping: u8,
        buildhasher: S,
    ) -> PrvCuckooMap<K, V, S, M> {
        let mut ret = PrvCuckooMap {
            key: std::marker::PhantomData,
            value: std::marker::PhantomData,
            buildhasher,
            branches: Vec::<M>::with_capacity(num_branches as usize),
        };
        for i in 0..num_branches {
            ret.branches.push(M::new(
                bucket_size,
                log2_branch_size,
                bucket_grouping,
                i as u32 * log2_branch_size as u32,
            ));
        }
        ret
    }
    pub fn hash_key(&self, key: K) -> Finish2 {
        let mut hasher = self.buildhasher.build_hasher();
        key.hash(&mut hasher);
        hasher.finish2()
    }

    pub fn capacity(&self) -> usize {
        let mut ret = 0;
        for b in self.branches.iter() {
            ret += b.capacity();
        }
        ret
    }

    pub fn length(&self) -> usize {
        let mut ret = 0;
        for b in self.branches.iter() {
            ret += b.length();
        }
        ret
    }

    fn get_mut_int(&mut self, f2: &Finish2) -> Option<&mut Entry<V>> {
        let mut ret = None;
        for b in self.branches.iter_mut() {
            match b.get(f2) {
                None => {}
                Some(x) => {
                    ret = Some(x);
                    break;
                }
            }
        }
        ret
    }

    pub fn get_mut(&mut self, key: K) -> Option<&mut Entry<V>> {
        self.get_mut_int(&self.hash_key(key))
    }

    pub fn get(&mut self, key: K) -> Option<&Entry<V>> {
        match self.get_mut_int(&self.hash_key(key)) {
            None => None,
            Some(x) => Some(&*x),
        }
    }

    pub fn insert(&mut self, key: K, value: V, depth: u32) -> Result<Option<Entry<V>>, Entry<V>> {
        let hashes = self.hash_key(key);
        let mut ret: Result<Option<Entry<V>>, Entry<V>> = Ok(None);
        let mut b_not_found = true;
        if let Some(x) = self.get_mut_int(&hashes) {
            ret = Ok(Some(x.clone()));
            b_not_found = false;
        }

        let mut entry = Entry { hashes, value };
        if b_not_found {
            for _i in 0..depth {
                for b in self.branches.iter_mut() {
                    match b.insert(entry) {
                        None => return Ok(None),
                        Some(x) => entry = x,
                    }
                }
            }
            ret = Err(entry);
        }
        ret
    }
}

// ___________________________________________________________________________________________________________

fn eq_fin2(f1: &Finish2, f2: &Finish2) -> bool {
    f1.as_u8_nonzero == f2.as_u8_nonzero && f1.hash == f2.hash
}

fn hash_loc(hash: u64, mask: u64, hash_rotator: u32) -> usize {
    (hash.rotate_right(hash_rotator) & mask) as usize
}

impl<V> EntryTrait for Entry<V>
where
    V: Default + Clone,
{
    //
    type V = V;
    fn hashes(&self) -> &Finish2 {
        &self.hashes
    }

    fn value(&mut self) -> &mut Self::V {
        &mut self.value
    }
}

impl<V> BranchEntryTrait for Entry<V>
where
    V: Default + Clone,
{
    type E = Self;
    fn new(_entrycap: u8) -> Self {
        Entry::default()
    }

    fn ptr_displace(&self) -> u8 {
        0
    }

    fn is_full(&self) -> bool {
        self.hashes().as_u8_nonzero != 0
    }

    fn clear(&mut self) {
        self.hashes = Finish2::default();
        self.value = V::default();
    }

    fn remove(&mut self, hashes: &Finish2) -> Option<Self::E> {
        if eq_fin2(self.hashes(), hashes) {
            let tmp = self.clone();
            self.clear();
            Some(tmp)
        } else {
            None
        }
    }

    fn get(&mut self, hashes: &Finish2) -> Option<&mut Self::E> {
        if eq_fin2(self.hashes(), hashes) {
            Some(self)
        } else {
            None
        }
    }

    fn insert(&mut self, entry: Self::E) -> Option<Self::E> {
        if self.hashes.as_u8_nonzero == 0 {
            *self = entry;
            None
        } else {
            let tmp = self.clone();
            *self = entry;
            Some(tmp)
        }
    }
}

impl<V> BranchEntryTrait for Bucket<V>
where
    V: Default + Clone,
{
    type E = Entry<V>;

    fn new(entrycap: u8) -> Self {
        Bucket {
            displace_index: 0,
            meta: vec![0; entrycap as usize],
            data: vec![Self::E::default(); entrycap as usize],
        }
    }

    fn ptr_displace(&self) -> u8 {
        self.displace_index
    }
    fn clear(&mut self) {
        self.displace_index = 0;
        for (i, e) in self.meta.iter_mut().enumerate() {
            *e = 0;
            self.data[i] = Entry::default();
        }
    }

    fn remove(&mut self, hashes: &Finish2) -> Option<Self::E> {
        let mut tmp = Self::E::default();
        let mut empty_ind: usize = u8::MAX as usize + 1;
        for (i, e) in self.meta.iter_mut().enumerate() {
            if *e == hashes.as_u8_nonzero as u8 && eq_fin2(self.data[i].hashes(), hashes) {
                tmp = self.data[i].clone();
                self.data[i] = Self::E::default();
                *e = 0;
                empty_ind = i;
                break;
            }
        }
        if empty_ind < u8::MAX as usize + 1 {
            for i in empty_ind..(empty_ind + self.meta.len()) {
                let i = i % self.meta.len();
                let j = (i + 1) % self.meta.len();
                if j == self.ptr_displace() as usize {
                    break;
                } else {
                    self.meta.swap(i, j);
                    self.data.swap(i, j);
                }
            }

            self.displace_index =
                ((self.displace_index as usize + (self.meta.len() - 1)) % self.meta.len()) as u8;
            Some(tmp)
        } else {
            None
        }
    }

    fn is_full(&self) -> bool {
        for e in self.meta.iter() {
            if *e == 0 {
                return false;
            }
        }
        true
    }

    fn get(&mut self, hashes: &Finish2) -> Option<&mut Self::E> {
        for (i, e) in self.meta.iter_mut().enumerate() {
            if *e == hashes.as_u8_nonzero as u8 && eq_fin2(self.data[i].hashes(), hashes) {
                return Some(&mut self.data[i]);
            }
        }
        None
    }

    fn insert(&mut self, entry: Self::E) -> Option<Self::E> {
        if self.meta[self.displace_index as usize] == 0 {
            self.meta[self.displace_index as usize] = entry.hashes().as_u8_nonzero as u8;
            self.data[self.displace_index as usize] = entry;
            self.displace_index = ((self.displace_index as usize + 1) % self.meta.len()) as u8;
            None
        } else {
            let tmp = self.data[self.displace_index as usize].clone();
            self.meta[self.displace_index as usize] = entry.hashes().as_u8_nonzero as u8;
            self.data[self.displace_index as usize] = entry;
            self.displace_index = ((self.displace_index as usize + 1) % self.meta.len()) as u8;
            Some(tmp)
        }
    }
}

impl<V, B> BranchTrait for BranchWithBucketGroups<V, B>
where
    V: Default + Clone,
    B: BranchEntryTrait<E = Entry<V>>,
{
    type E = Entry<V>;
    type BE = B;

    fn new(
        bucket_entrycap: u8,
        log2_bucketcap: u8,
        bucket_grouping: u8,
        hash_rotator: u32,
    ) -> Self {
        let mut ret = Self {
            hash_rotator,
            mask: u64::pow(2, log2_bucketcap as u32) - 1,
            bucket_grouping,
            bucket_entrycap,
            cap_buckets: usize::pow(2, log2_bucketcap as u32),
            length: 0,
            data: Vec::with_capacity(usize::pow(2, log2_bucketcap as u32)),
            value_type: std::marker::PhantomData,
        };
        for _i in 0..ret.cap_buckets {
            ret.data.push(Self::BE::new(bucket_entrycap));
        }
        ret
    }

    fn capacity(&self) -> usize {
        self.cap_buckets * self.bucket_entrycap as usize
    }

    fn length(&self) -> usize {
        self.length
    }

    fn clear(&mut self) {
        for bucket in self.data.iter_mut() {
            bucket.clear();
        }
        self.length = 0;
    }

    fn remove(&mut self, hashes: &Finish2) -> Option<Self::E> {
        for i in 0..self.bucket_grouping as usize + 1 {
            match self.data
                [(hash_loc(hashes.hash, self.mask, self.hash_rotator) + i) % self.cap_buckets]
                .remove(hashes)
            {
                None => {}
                Some(x) => {
                    self.length -= 1;
                    return Some(x);
                }
            }
        }
        None
    }
    fn get(&mut self, hashes: &Finish2) -> Option<&mut Self::E> {
        for i in 0..self.bucket_grouping as usize + 1 {
            match self.data
                [(hash_loc(hashes.hash, self.mask, self.hash_rotator) + i) % self.cap_buckets]
                .get(hashes)
            {
                None => {}
                Some(x) => unsafe {
                    return Some(&mut *(x as *mut Self::E));
                },
            }
        }
        None
    }

    fn insert(&mut self, entry: Self::E) -> Option<Self::E> {
        let hl = hash_loc(entry.hashes().hash, self.mask, self.hash_rotator);
        let mut ind_insert = hl;
        for i in hl..(hl + self.bucket_grouping as usize + 1) {
            let i = i % self.cap_buckets;
            if (self.data[i].ptr_displace() < self.data[ind_insert].ptr_displace())
                | (self.data[ind_insert].ptr_displace() == 0
                    && self.data[i].ptr_displace() >= self.bucket_entrycap - 1)
            {
                ind_insert = i;
            }
            if !self.data[i].is_full() {
                ind_insert = i;
                self.length += 1;
                break;
            }
        }
        self.data[ind_insert].insert(entry)
    }
}

impl<V, B> BranchTrait for BranchWithoutBucketGroups<V, B>
where
    V: Default + Clone,
    B: BranchEntryTrait<E = Entry<V>>,
{
    type E = Entry<V>;
    type BE = B;

    fn new(
        bucket_entrycap: u8,
        log2_bucketcap: u8,
        _bucket_grouping: u8,
        hash_rotator: u32,
    ) -> Self {
        let mut ret = Self {
            hash_rotator,
            mask: u64::pow(2, log2_bucketcap as u32) - 1,
            bucket_entrycap,
            cap_buckets: usize::pow(2, log2_bucketcap as u32),
            length: 0,
            data: Vec::with_capacity(usize::pow(2, log2_bucketcap as u32)),
            value_type: std::marker::PhantomData,
        };
        for _i in 0..ret.cap_buckets {
            ret.data.push(Self::BE::new(bucket_entrycap));
        }
        ret
    }

    fn capacity(&self) -> usize {
        self.cap_buckets * self.bucket_entrycap as usize
    }

    fn length(&self) -> usize {
        self.length
    }

    fn clear(&mut self) {
        for bucket in self.data.iter_mut() {
            bucket.clear();
        }
        self.length = 0;
    }

    fn remove(&mut self, hashes: &Finish2) -> Option<Self::E> {
        match self.data[hash_loc(hashes.hash, self.mask, self.hash_rotator)].remove(hashes) {
            None => None,
            Some(x) => {
                self.length -= 1;
                Some(x)
            }
        }
    }

    fn get(&mut self, hashes: &Finish2) -> Option<&mut Self::E> {
        match self.data[hash_loc(hashes.hash, self.mask, self.hash_rotator)].get(hashes) {
            None => None,
            Some(x) => Some(x),
        }
    }

    fn insert(&mut self, entry: Self::E) -> Option<Self::E> {
        self.data[hash_loc(entry.hashes().hash, self.mask, self.hash_rotator)].insert(entry)
    }
}

// ##############################################################################################################

#[cfg(test)]
mod tests {
    use super::CuckooMap;

    #[test]
    fn insert_then_get() {
        let mut cm = CuckooMap::<u64, u64>::new_with_buckets(3, 16, 16);

        let test_size = (cm.capacity() as f64 * 0.98) as u64;

        println!("Starting inserting");

        let bcmi = std::time::Instant::now();

        for i in 0..test_size {
            match cm.insert(i, i, 200) {
                Ok(_) => {}
                Err(x) => panic!("Key displaced: {}", x.value),
            }
        }

        let bcmif = bcmi.elapsed();

        println!("Finished inserting, starting getting");

        let bcmg = std::time::Instant::now();

        for i in 0..test_size {
            if cm.get(i).is_none() {
                panic!("Key not found: {}", i);
            }
        }

        let bcmgf = bcmg.elapsed();

        println!("Finished getting, Results:");
        println!(
            "M In/s CM: {}",
            test_size as f64 / bcmif.as_secs_f64() / 1_000_000.0
        );
        println!(
            "M Lu/s CM: {}",
            test_size as f64 / bcmgf.as_secs_f64() / 1_000_000.0
        );
    }
}
