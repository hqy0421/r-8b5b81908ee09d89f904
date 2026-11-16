import pickle
import unittest

import numpy as np
import scipy as sp
import scipy.sparse

from datasketch.weighted_minhash import WeightedMinHash, WeightedMinHashGenerator


def _random_matrix(num_rows, dim, seed, zero_rows=()):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 6, size=(num_rows, dim)).astype(np.float32)
    X = X * rng.uniform(0.5, 3.0, size=X.shape).astype(np.float32)
    for row in zero_rows:
        X[row] = 0.0
    return X


class TestWeightedMinHash(unittest.TestCase):
    def test_pickle(self):
        mg = WeightedMinHashGenerator(4, 10, 1)
        m = mg.minhash([1, 2, 3, 4])
        p = pickle.loads(pickle.dumps(m))
        self.assertEqual(p.seed, m.seed)
        self.assertTrue(np.array_equal(p.hashvalues, m.hashvalues))


class TestWeightedMinHashGenerator(unittest.TestCase):
    def test_init(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        self.assertEqual(len(mg.rs), 4)
        self.assertEqual(len(mg.ln_cs), 4)
        self.assertEqual(len(mg.betas), 4)
        self.assertEqual(mg.seed, 1)
        self.assertEqual(mg.sample_size, 4)

    def test_minhash(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m = mg.minhash([1, 3])
        self.assertIsInstance(m, WeightedMinHash)
        self.assertEqual(len(m.hashvalues), 4)
        self.assertEqual(len(m), 4)
        self.assertTrue(m.hashvalues.dtype == int)

    def test_minhash_does_not_mutate_float32_input(self):
        mg = WeightedMinHashGenerator(3, 4, 1)
        v = np.array([1, 0, 3], dtype=np.float32)
        expected = v.copy()
        mg.minhash(v)
        np.testing.assert_array_equal(v, expected)

    def test_minhash_many_dense_onerow(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m_list = mg.minhash_many(np.array([1, 3]).reshape(1, 2))
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 1)
        self.assertIsInstance(m_list[0], WeightedMinHash)
        self.assertEqual(len(m_list[0].hashvalues), 4)
        self.assertEqual(len(m_list[0]), 4)
        self.assertTrue(m_list[0].hashvalues.dtype == int)

    def test_minhash_many_dense_tworows(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m_list = mg.minhash_many(np.array([[1, 3], [1, 3]]))
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 2)
        for m in m_list:
            self.assertIsInstance(m, WeightedMinHash)
            self.assertEqual(len(m.hashvalues), 4)
            self.assertEqual(len(m), 4)
            self.assertTrue(m.hashvalues.dtype == int)

    def test_minhash_many_dense_tworows_with_null(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m_list = mg.minhash_many(np.array([[1, 3], [0, 0]]))
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 2)

        m = m_list[0]
        self.assertIsInstance(m, WeightedMinHash)
        self.assertEqual(len(m.hashvalues), 4)
        self.assertEqual(len(m), 4)
        self.assertTrue(m.hashvalues.dtype == int)

        self.assertIs(m_list[1], None)

    def test_minhash_many_sparse_onerow(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        X = sp.sparse.csr_matrix(np.array([1, 3]).reshape(1, 2))
        m_list = mg.minhash_many(X)
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 1)
        self.assertIsInstance(m_list[0], WeightedMinHash)
        self.assertEqual(len(m_list[0].hashvalues), 4)
        self.assertEqual(len(m_list[0]), 4)
        self.assertTrue(m_list[0].hashvalues.dtype == int)

    def test_minhash_many_sparse_tworows(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        X = sp.sparse.csr_matrix(np.array([[1, 3], [1, 3]]))
        m_list = mg.minhash_many(X)
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 2)
        for m in m_list:
            self.assertIsInstance(m, WeightedMinHash)
            self.assertEqual(len(m.hashvalues), 4)
            self.assertEqual(len(m), 4)
            self.assertTrue(m.hashvalues.dtype == int)

    def test_minhash_many_sparse_tworows_with_null(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        X = sp.sparse.csr_matrix(np.array([[1, 3], [0, 0]]))
        m_list = mg.minhash_many(X)
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 2)

        m = m_list[0]
        self.assertIsInstance(m, WeightedMinHash)
        self.assertEqual(len(m.hashvalues), 4)
        self.assertEqual(len(m), 4)
        self.assertTrue(m.hashvalues.dtype == int)

        self.assertIs(m_list[1], None)


class TestWeightedMinHashGeneratorBatch(unittest.TestCase):
    def test_batch_sparse_matches_single_row(self):
        dim, sample_size, num_rows = 40, 32, 25
        mg = WeightedMinHashGenerator(dim, sample_size, seed=11)
        X = _random_matrix(num_rows, dim, seed=0, zero_rows=(3, 17))
        sparse_X = sp.sparse.csr_matrix(X)

        batch = mg.minhash_batch(sparse_X)

        self.assertIsInstance(batch, list)
        self.assertEqual(len(batch), num_rows)
        for row in range(num_rows):
            with self.subTest(row=row):
                if row in (3, 17):
                    self.assertIsNone(batch[row])
                else:
                    single = mg.minhash(X[row].copy())
                    self.assertIsInstance(batch[row], WeightedMinHash)
                    np.testing.assert_array_equal(batch[row].hashvalues, single.hashvalues)
                    self.assertEqual(batch[row].seed, single.seed)

    def test_batch_dense_matches_single_row(self):
        dim, sample_size, num_rows = 20, 16, 10
        mg = WeightedMinHashGenerator(dim, sample_size, seed=5)
        X = _random_matrix(num_rows, dim, seed=1, zero_rows=(0, 9))

        batch = mg.minhash_batch(X)

        self.assertEqual(len(batch), num_rows)
        for row in range(num_rows):
            with self.subTest(row=row):
                if row in (0, 9):
                    self.assertIsNone(batch[row])
                else:
                    single = mg.minhash(X[row].copy())
                    np.testing.assert_array_equal(batch[row].hashvalues, single.hashvalues)

    def test_batch_dense_and_sparse_agree(self):
        dim, sample_size, num_rows = 20, 16, 8
        mg = WeightedMinHashGenerator(dim, sample_size, seed=5)
        X = _random_matrix(num_rows, dim, seed=2, zero_rows=(4,))

        dense_batch = mg.minhash_batch(X)
        sparse_batch = mg.minhash_batch(sp.sparse.csr_matrix(X))

        for a, b in zip(dense_batch, sparse_batch):
            if a is None:
                self.assertIsNone(b)
            else:
                np.testing.assert_array_equal(a.hashvalues, b.hashvalues)

    def test_batch_single_row(self):
        dim, sample_size = 15, 8
        mg = WeightedMinHashGenerator(dim, sample_size, seed=2)
        v = np.array([0, 1, 0, 4, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 7], dtype=np.float32)

        batch = mg.minhash_batch(sp.sparse.csr_matrix(v.reshape(1, dim)))

        self.assertEqual(len(batch), 1)
        single = mg.minhash(v.copy())
        np.testing.assert_array_equal(batch[0].hashvalues, single.hashvalues)

    def test_batch_all_zero_rows_return_none_in_place(self):
        dim, sample_size = 10, 8
        mg = WeightedMinHashGenerator(dim, sample_size, seed=2)
        X = sp.sparse.csr_matrix(np.zeros((4, dim), dtype=np.float32))

        batch = mg.minhash_batch(X)

        self.assertEqual(batch, [None, None, None, None])

    def test_batch_empty_matrix(self):
        dim = 10
        mg = WeightedMinHashGenerator(dim, 8, seed=2)

        self.assertEqual(mg.minhash_batch(np.zeros((0, dim), dtype=np.float32)), [])
        self.assertEqual(mg.minhash_batch(sp.sparse.csr_matrix((0, dim), dtype=np.float32)), [])

    def test_batch_does_not_mutate_input(self):
        dim, sample_size = 12, 8
        mg = WeightedMinHashGenerator(dim, sample_size, seed=2)

        # Sparse input with unsorted columns, duplicate coordinates and an
        # explicit zero must be left untouched.
        data = np.array([2.0, 3.0, 1.0, 0.0], dtype=np.float32)
        indices = np.array([9, 2, 9, 5])
        indptr = np.array([0, 4])
        sparse_X = sp.sparse.csr_matrix((data, indices, indptr), shape=(1, dim))
        sparse_snapshot = (
            sparse_X.data.copy(),
            sparse_X.indices.copy(),
            sparse_X.indptr.copy(),
        )

        mg.minhash_batch(sparse_X)

        np.testing.assert_array_equal(sparse_X.data, sparse_snapshot[0])
        np.testing.assert_array_equal(sparse_X.indices, sparse_snapshot[1])
        np.testing.assert_array_equal(sparse_X.indptr, sparse_snapshot[2])

        # Dense input must be left untouched as well.
        dense_X = _random_matrix(3, dim, seed=3, zero_rows=(1,))
        dense_snapshot = dense_X.copy()
        mg.minhash_batch(dense_X)
        np.testing.assert_array_equal(dense_X, dense_snapshot)

    def test_batch_handles_unsorted_duplicate_and_explicit_zero(self):
        dim, sample_size = 12, 8
        mg = WeightedMinHashGenerator(dim, sample_size, seed=2)

        # Coordinate 9 appears twice (2 and 1 -> summed to 3), coordinate 5
        # holds an explicit zero (dropped), columns are unsorted.
        data = np.array([2.0, 3.0, 1.0, 0.0], dtype=np.float32)
        indices = np.array([9, 2, 9, 5])
        indptr = np.array([0, 4])
        sparse_X = sp.sparse.csr_matrix((data, indices, indptr), shape=(1, dim))

        batch = mg.minhash_batch(sparse_X)

        dense = np.zeros(dim, dtype=np.float32)
        dense[9] = 3.0
        dense[2] = 3.0
        single = mg.minhash(dense)
        np.testing.assert_array_equal(batch[0].hashvalues, single.hashvalues)

    def test_batch_duplicate_coordinates_summed_across_formats(self):
        dim, sample_size = 12, 8
        mg = WeightedMinHashGenerator(dim, sample_size, seed=2)

        coo = sp.sparse.coo_matrix(
            (
                np.array([1.0, 3.0, 2.0], dtype=np.float32),
                (np.array([0, 0, 0]), np.array([1, 1, 5])),
            ),
            shape=(1, dim),
        )
        batch = mg.minhash_batch(coo)

        dense = np.zeros(dim, dtype=np.float32)
        dense[1] = 4.0
        dense[5] = 2.0
        single = mg.minhash(dense)
        np.testing.assert_array_equal(batch[0].hashvalues, single.hashvalues)

    def test_batch_does_not_mutate_generator_parameters(self):
        dim, sample_size = 10, 8
        mg = WeightedMinHashGenerator(dim, sample_size, seed=2)
        rs, betas, ln_cs = mg.rs.copy(), mg.betas.copy(), mg.ln_cs.copy()
        X = _random_matrix(5, dim, seed=4, zero_rows=(2,))

        mg.minhash_batch(sp.sparse.csr_matrix(X))

        np.testing.assert_array_equal(mg.rs, rs)
        np.testing.assert_array_equal(mg.betas, betas)
        np.testing.assert_array_equal(mg.ln_cs, ln_cs)

    def test_batch_rejects_illegal_weights(self):
        dim, sample_size = 10, 8
        mg = WeightedMinHashGenerator(dim, sample_size, seed=2)

        def row_with(value):
            v = np.zeros(dim, dtype=np.float32)
            v[0] = 1.0
            v[1] = value
            return v.reshape(1, dim)

        for value in (-1.0, np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                dense = row_with(np.float32(value))
                with self.assertRaises(ValueError):
                    mg.minhash_batch(dense)
                with self.assertRaises(ValueError):
                    mg.minhash_batch(sp.sparse.csr_matrix(dense))

    def test_batch_negative_zero_is_treated_as_zero(self):
        dim, sample_size = 10, 8
        mg = WeightedMinHashGenerator(dim, sample_size, seed=2)
        v = np.array([-0.0, 2.0, 0.0, 4.0] + [0.0] * (dim - 4), dtype=np.float32)

        batch = mg.minhash_batch(sp.sparse.csr_matrix(v.reshape(1, dim)))
        single = mg.minhash(v.copy())

        np.testing.assert_array_equal(batch[0].hashvalues, single.hashvalues)

    def test_batch_input_validation(self):
        dim, sample_size = 10, 8
        mg = WeightedMinHashGenerator(dim, sample_size, seed=2)

        with self.assertRaises(TypeError):
            mg.minhash_batch([[1, 2, 3]])
        with self.assertRaises(ValueError):
            mg.minhash_batch(np.zeros(dim, dtype=np.float32))
        with self.assertRaises(ValueError):
            mg.minhash_batch(np.zeros((2, dim - 1), dtype=np.float32))

    def test_batch_hashvalues_dtype_and_shape(self):
        dim, sample_size = 10, 8
        mg = WeightedMinHashGenerator(dim, sample_size, seed=2)
        X = _random_matrix(3, dim, seed=5)

        batch = mg.minhash_batch(sp.sparse.csr_matrix(X))

        for m in batch:
            self.assertEqual(m.hashvalues.shape, (sample_size, 2))
            self.assertTrue(np.issubdtype(m.hashvalues.dtype, np.integer))

    def test_minhash_many_experimental_behavior_preserved(self):
        # The pre-existing experimental batch path keeps its old contract:
        # positional Nones for empty rows and WeightedMinHash otherwise.
        dim, sample_size = 6, 4
        mg = WeightedMinHashGenerator(dim, sample_size, seed=1)
        X = np.array([[1, 3, 0, 0, 2, 0], [0, 0, 0, 0, 0, 0]], dtype=np.float32)

        for factory in (np.asarray, sp.sparse.csr_matrix):
            with self.subTest(factory=factory.__name__):
                result = mg.minhash_many(factory(X))
                self.assertEqual(len(result), 2)
                self.assertIsInstance(result[0], WeightedMinHash)
                self.assertIsNone(result[1])


if __name__ == "__main__":
    unittest.main()
