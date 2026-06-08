# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for src.core.ternary - TernarySpace singleton."""

import pytest
import torch

from src.core.ternary import TERNARY, TernarySpace


class TestTernarySpaceConstants:
    def test_n_operations_is_3_to_the_9th(self):
        assert TernarySpace.N_OPERATIONS == 19683
        assert TernarySpace.N_OPERATIONS == 3**9

    def test_n_digits_is_nine(self):
        assert TernarySpace.N_DIGITS == 9

    def test_max_valuation_is_nine(self):
        assert TernarySpace.MAX_VALUATION == 9

    def test_ternary_values(self):
        assert TernarySpace.TERNARY_VALUES == (-1, 0, 1)

    def test_singleton_is_ternary_space(self):
        assert isinstance(TERNARY, TernarySpace)


class TestValuation:
    def test_zero_valuation_is_max(self):
        v = TERNARY.valuation(torch.tensor([0]))
        assert v[0].item() == TernarySpace.MAX_VALUATION

    def test_one_is_not_divisible_by_three(self):
        v = TERNARY.valuation(torch.tensor([1]))
        assert v[0].item() == 0

    def test_two_is_not_divisible_by_three(self):
        v = TERNARY.valuation(torch.tensor([2]))
        assert v[0].item() == 0

    def test_three_has_valuation_one(self):
        v = TERNARY.valuation(torch.tensor([3]))
        assert v[0].item() == 1

    def test_nine_has_valuation_two(self):
        v = TERNARY.valuation(torch.tensor([9]))
        assert v[0].item() == 2

    def test_twenty_seven_has_valuation_three(self):
        v = TERNARY.valuation(torch.tensor([27]))
        assert v[0].item() == 3

    def test_batch_valuation(self):
        indices = torch.tensor([1, 3, 9, 27])
        v = TERNARY.valuation(indices)
        assert v.tolist() == [0, 1, 2, 3]

    def test_output_shape_preserved(self):
        indices = torch.arange(100)
        v = TERNARY.valuation(indices)
        assert v.shape == indices.shape

    def test_output_dtype_is_long(self):
        v = TERNARY.valuation(torch.tensor([9]))
        assert v.dtype == torch.long

    def test_clamping_out_of_range(self):
        # Values outside [0, N_OPERATIONS-1] are clamped, should not error
        v = TERNARY.valuation(torch.tensor([99999]))
        assert v.shape == (1,)

    @pytest.mark.parametrize("index,expected", [(0, 9), (1, 0), (3, 1), (9, 2), (27, 3), (81, 4)])
    def test_known_valuations(self, index, expected):
        v = TERNARY.valuation(torch.tensor([index]))
        assert v[0].item() == expected


class TestDistance:
    def test_same_index_is_zero(self):
        d = TERNARY.distance(torch.tensor([0]), torch.tensor([0]))
        assert d[0].item() == 0.0

    def test_same_nonzero_index_is_zero(self):
        d = TERNARY.distance(torch.tensor([42]), torch.tensor([42]))
        assert d[0].item() == 0.0

    def test_symmetric(self):
        a = torch.tensor([1, 5, 10])
        b = torch.tensor([10, 50, 100])
        d_ab = TERNARY.distance(a, b)
        d_ba = TERNARY.distance(b, a)
        assert torch.allclose(d_ab, d_ba)

    def test_non_negative(self):
        a = torch.arange(0, 50)
        b = torch.arange(50, 100)
        d = TERNARY.distance(a, b)
        assert (d >= 0.0).all()

    def test_at_most_one(self):
        a = torch.arange(0, 100)
        b = torch.arange(100, 200)
        d = TERNARY.distance(a, b)
        assert (d <= 1.0).all()

    def test_batch_preserves_shape(self):
        a = torch.arange(20)
        b = torch.arange(20, 40)
        d = TERNARY.distance(a, b)
        assert d.shape == a.shape

    def test_ultrametric_property(self):
        # d(a, c) ≤ max(d(a, b), d(b, c))
        a = torch.tensor([0, 0, 0])
        b = torch.tensor([9, 9, 3])
        c = torch.tensor([27, 18, 27])
        d_ac = TERNARY.distance(a, c)
        d_ab = TERNARY.distance(a, b)
        d_bc = TERNARY.distance(b, c)
        max_side = torch.maximum(d_ab, d_bc)
        assert (d_ac <= max_side + 1e-6).all()


class TestToTernary:
    def test_output_shape(self):
        indices = torch.arange(10)
        ternary = TERNARY.to_ternary(indices)
        assert ternary.shape == (10, TernarySpace.N_DIGITS)

    def test_single_index_shape(self):
        ternary = TERNARY.to_ternary(torch.tensor([5]))
        assert ternary.shape == (1, TernarySpace.N_DIGITS)

    def test_values_in_valid_set(self):
        indices = torch.arange(100)
        ternary = TERNARY.to_ternary(indices)
        unique = set(ternary.reshape(-1).tolist())
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_all_indices_produce_valid_ternary(self):
        all_idx = TERNARY.all_indices()
        ternary = TERNARY.to_ternary(all_idx)
        assert ternary.shape == (TernarySpace.N_OPERATIONS, TernarySpace.N_DIGITS)


class TestFromTernary:
    def test_roundtrip_to_and_from_ternary(self):
        indices = torch.arange(0, 200)
        ternary = TERNARY.to_ternary(indices)
        recovered = TERNARY.from_ternary(ternary)
        assert torch.equal(indices, recovered)

    def test_single_roundtrip(self):
        idx = torch.tensor([42])
        recovered = TERNARY.from_ternary(TERNARY.to_ternary(idx))
        assert recovered[0].item() == 42

    def test_full_roundtrip(self):
        all_idx = TERNARY.all_indices()
        ternary = TERNARY.to_ternary(all_idx)
        recovered = TERNARY.from_ternary(ternary)
        assert torch.equal(all_idx, recovered)


class TestValidityChecks:
    def test_valid_indices_in_range(self):
        indices = torch.tensor([0, 100, 19682])
        assert TERNARY.is_valid_index(indices).all()

    def test_invalid_index_negative(self):
        indices = torch.tensor([-1])
        assert not TERNARY.is_valid_index(indices).all()

    def test_invalid_index_too_large(self):
        indices = torch.tensor([19683])
        assert not TERNARY.is_valid_index(indices).all()

    def test_valid_ternary(self):
        ternary = torch.tensor([[-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]])
        assert TERNARY.is_valid_ternary(ternary).all()

    def test_invalid_ternary(self):
        ternary = torch.tensor([[2.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]])
        assert not TERNARY.is_valid_ternary(ternary).all()


class TestHelpers:
    def test_all_indices_length(self):
        idx = TERNARY.all_indices()
        assert len(idx) == TernarySpace.N_OPERATIONS

    def test_all_indices_range(self):
        idx = TERNARY.all_indices()
        assert idx[0].item() == 0
        assert idx[-1].item() == TernarySpace.N_OPERATIONS - 1

    def test_sample_indices_shape(self):
        samples = TERNARY.sample_indices(100)
        assert samples.shape == (100,)

    def test_sample_indices_in_range(self):
        samples = TERNARY.sample_indices(500)
        assert TERNARY.is_valid_index(samples).all()
