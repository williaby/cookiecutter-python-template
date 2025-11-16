"""Comprehensive tests for secure_random.py coverage gaps - targeting 0% coverage functions."""

import pytest

from src.utils.secure_random import SecureRandom


class TestSecureRandomCoverageGaps:
    """Test SecureRandom methods with 0% coverage."""

    def test_secure_random_sample_method(self):
        """Test sample method with 0% coverage."""
        secure_rng = SecureRandom()

        # Test normal sampling
        population = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sample_size = 3

        result = secure_rng.sample(population, sample_size)

        # Verify sample properties
        assert len(result) == sample_size
        assert all(item in population for item in result)
        assert len(set(result)) == sample_size  # All elements should be unique

    def test_secure_random_sample_edge_cases(self):
        """Test sample method edge cases."""
        secure_rng = SecureRandom()

        # Test sampling all elements
        population = ["a", "b", "c"]
        result = secure_rng.sample(population, 3)
        assert len(result) == 3
        assert set(result) == set(population)

        # Test sampling zero elements
        result = secure_rng.sample(population, 0)
        assert result == []

        # Test sampling one element
        result = secure_rng.sample(population, 1)
        assert len(result) == 1
        assert result[0] in population

    def test_secure_random_sample_error_handling(self):
        """Test sample method error conditions."""
        secure_rng = SecureRandom()

        population = [1, 2, 3]

        # Test k > population size
        with pytest.raises(ValueError, match="Sample size cannot exceed population size"):
            secure_rng.sample(population, 5)

        # Test empty population with non-zero k
        with pytest.raises(ValueError, match="Sample size cannot exceed population size"):
            secure_rng.sample([], 1)

    def test_secure_random_shuffle_method(self):
        """Test shuffle method with 0% coverage."""
        secure_rng = SecureRandom()

        # Test normal shuffling
        original_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        test_list = original_list.copy()

        secure_rng.shuffle(test_list)

        # Verify shuffle properties
        assert len(test_list) == len(original_list)
        assert set(test_list) == set(original_list)  # Same elements
        # Note: We can't guarantee the order changed due to randomness

    def test_secure_random_shuffle_edge_cases(self):
        """Test shuffle method edge cases."""
        secure_rng = SecureRandom()

        # Test empty list
        empty_list = []
        secure_rng.shuffle(empty_list)
        assert empty_list == []

        # Test single element list
        single_list = [42]
        secure_rng.shuffle(single_list)
        assert single_list == [42]

        # Test two element list
        two_list = [1, 2]
        secure_rng.shuffle(two_list)
        assert len(two_list) == 2
        assert set(two_list) == {1, 2}

    def test_secure_random_shuffle_in_place_modification(self):
        """Test shuffle modifies list in place."""
        secure_rng = SecureRandom()

        original_list = [1, 2, 3, 4, 5]
        list_id = id(original_list)

        secure_rng.shuffle(original_list)

        # Verify same object (in-place modification)
        assert id(original_list) == list_id
        assert len(original_list) == 5
        assert set(original_list) == {1, 2, 3, 4, 5}

    def test_secure_random_sample_different_types(self):
        """Test sample method with different data types."""
        secure_rng = SecureRandom()

        # Test with strings
        string_population = ["apple", "banana", "cherry", "date"]
        result = secure_rng.sample(string_population, 2)
        assert len(result) == 2
        assert all(item in string_population for item in result)

        # Test with mixed types
        mixed_population = [1, "two", 3.0, None, True]
        result = secure_rng.sample(mixed_population, 3)
        assert len(result) == 3
        assert all(item in mixed_population for item in result)

    def test_secure_random_shuffle_different_types(self):
        """Test shuffle method with different data types."""
        secure_rng = SecureRandom()

        # Test with strings
        string_list = ["red", "green", "blue", "yellow"]
        original_set = set(string_list)
        secure_rng.shuffle(string_list)
        assert set(string_list) == original_set

        # Test with mixed types
        mixed_list = [1, "two", 3.0, None, True]
        original_set = set(mixed_list)
        secure_rng.shuffle(mixed_list)
        assert set(mixed_list) == original_set

    def test_secure_random_sample_reproducibility(self):
        """Test that sample produces different results (due to cryptographic randomness)."""
        secure_rng = SecureRandom()

        population = list(range(20))  # Large enough population
        sample_size = 5

        # Generate multiple samples
        samples = [secure_rng.sample(population, sample_size) for _ in range(10)]

        # Verify all samples are different (with high probability)
        unique_samples = {tuple(sample) for sample in samples}
        assert len(unique_samples) > 1  # Should be very likely with crypto randomness

    def test_secure_random_shuffle_reproducibility(self):
        """Test that shuffle produces different results (due to cryptographic randomness)."""
        secure_rng = SecureRandom()

        # Generate multiple shuffled lists
        original = list(range(10))
        shuffled_lists = []

        for _ in range(10):
            test_list = original.copy()
            secure_rng.shuffle(test_list)
            shuffled_lists.append(tuple(test_list))

        # Verify not all shuffles are identical (with high probability)
        unique_shuffles = set(shuffled_lists)
        assert len(unique_shuffles) > 1  # Should be very likely with crypto randomness


class TestSecureRandomSamplePerformance:
    """Test performance characteristics of sample method."""

    def test_secure_random_sample_large_population(self):
        """Test sample method with large population."""
        secure_rng = SecureRandom()

        # Test with large population
        large_population = list(range(10000))
        result = secure_rng.sample(large_population, 100)

        # Verify correct sampling
        assert len(result) == 100
        assert len(set(result)) == 100  # All unique
        assert all(0 <= item < 10000 for item in result)

    def test_secure_random_shuffle_large_list(self):
        """Test shuffle method with large list."""
        secure_rng = SecureRandom()

        # Test with large list
        large_list = list(range(1000))
        original_sum = sum(large_list)

        secure_rng.shuffle(large_list)

        # Verify shuffle correctness
        assert len(large_list) == 1000
        assert sum(large_list) == original_sum
        assert min(large_list) == 0
        assert max(large_list) == 999


class TestSecureRandomEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_secure_random_sample_boundary_cases(self):
        """Test sample method boundary cases."""
        secure_rng = SecureRandom()

        # Test sampling from tuple (immutable sequence)
        tuple_population = (10, 20, 30, 40, 50)
        result = secure_rng.sample(tuple_population, 3)
        assert len(result) == 3
        assert all(item in tuple_population for item in result)

        # Test sampling from string (sequence of characters)
        string_population = "abcdefgh"
        result = secure_rng.sample(string_population, 4)
        assert len(result) == 4
        assert all(char in string_population for char in result)

    def test_secure_random_shuffle_mutable_objects(self):
        """Test shuffle with lists containing mutable objects."""
        secure_rng = SecureRandom()

        # Test with lists containing lists
        nested_list = [[1, 2], [3, 4], [5, 6]]
        original_inner_lists = [sublist.copy() for sublist in nested_list]

        secure_rng.shuffle(nested_list)

        # Verify outer list shuffled but inner lists unchanged
        assert len(nested_list) == 3
        for sublist in nested_list:
            assert sublist in original_inner_lists


@pytest.mark.parametrize(
    ("population_size", "sample_size"),
    [
        (5, 1),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (10, 5),
        (100, 10),
    ],
)
def test_secure_random_sample_parametrized(population_size, sample_size):
    """Test sample method with various population and sample sizes."""
    secure_rng = SecureRandom()

    population = list(range(population_size))
    result = secure_rng.sample(population, sample_size)

    assert len(result) == sample_size
    assert len(set(result)) == sample_size  # All unique
    assert all(item in population for item in result)


@pytest.mark.parametrize("list_size", [1, 2, 5, 10, 50, 100])
def test_secure_random_shuffle_parametrized(list_size):
    """Test shuffle method with various list sizes."""
    secure_rng = SecureRandom()

    test_list = list(range(list_size))
    original_elements = set(test_list)

    secure_rng.shuffle(test_list)

    assert len(test_list) == list_size
    assert set(test_list) == original_elements
