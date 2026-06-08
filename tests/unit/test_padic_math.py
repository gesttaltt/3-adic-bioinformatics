# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for src.core.padic_math - pure p-adic math functions."""

import pytest

from src.core.padic_math import (
    DEFAULT_P,
    PADIC_INFINITY_INT,
    PAdicShiftResult,
    compute_goldilocks_score,
    is_in_goldilocks_zone,
    padic_digits,
    padic_distance,
    padic_norm,
    padic_shift,
    padic_valuation,
)


class TestPadicValuation:
    def test_zero_returns_infinity_sentinel(self):
        assert padic_valuation(0) == PADIC_INFINITY_INT

    def test_power_of_three(self):
        assert padic_valuation(3) == 1
        assert padic_valuation(9) == 2  # 9 = 3^2
        assert padic_valuation(27) == 3  # 27 = 3^3
        assert padic_valuation(81) == 4

    def test_not_divisible_by_three(self):
        assert padic_valuation(1) == 0
        assert padic_valuation(2) == 0
        assert padic_valuation(5) == 0
        assert padic_valuation(7) == 0

    def test_mixed_factors(self):
        assert padic_valuation(6) == 1  # 6 = 2 * 3
        assert padic_valuation(18) == 2  # 18 = 2 * 3^2
        assert padic_valuation(54) == 3  # 54 = 2 * 3^3

    def test_negative_input_uses_absolute_value(self):
        assert padic_valuation(-9) == padic_valuation(9)
        assert padic_valuation(-3) == padic_valuation(3)

    def test_custom_prime_two(self):
        assert padic_valuation(8, p=2) == 3  # 8 = 2^3
        assert padic_valuation(4, p=2) == 2
        assert padic_valuation(6, p=2) == 1  # 6 = 2 * 3

    def test_default_prime_is_three(self):
        assert DEFAULT_P == 3


class TestPadicNorm:
    def test_zero_returns_zero(self):
        assert padic_norm(0) == 0.0

    def test_not_divisible_returns_one(self):
        assert padic_norm(1) == 1.0
        assert padic_norm(2) == 1.0
        assert padic_norm(5) == 1.0

    def test_power_of_three(self):
        assert abs(padic_norm(3) - 1 / 3) < 1e-10
        assert abs(padic_norm(9) - 1 / 9) < 1e-10
        assert abs(padic_norm(27) - 1 / 27) < 1e-10

    def test_smaller_valuation_means_larger_norm(self):
        # v_3(3) = 1, v_3(9) = 2 → |3|_3 > |9|_3
        assert padic_norm(3) > padic_norm(9)

    def test_ultrametric_inequality(self):
        # |a + b|_p ≤ max(|a|_p, |b|_p)  (strong triangle inequality)
        # 3 + 9 = 12, |12|_3 = |3*4|_3 = 1/3 ≤ max(|3|_3, |9|_3) = 1/3
        assert padic_norm(12) <= max(padic_norm(3), padic_norm(9)) + 1e-10


class TestPadicDistance:
    def test_equal_elements_zero_distance(self):
        assert padic_distance(0, 0) == 0.0
        assert padic_distance(5, 5) == 0.0
        assert padic_distance(100, 100) == 0.0

    def test_symmetric(self):
        assert padic_distance(0, 9) == padic_distance(9, 0)
        assert padic_distance(3, 15) == padic_distance(15, 3)

    def test_ultrametric_strong_triangle(self):
        # d(a, c) ≤ max(d(a, b), d(b, c))
        a, b, c = 0, 9, 27
        assert padic_distance(a, c) <= max(padic_distance(a, b), padic_distance(b, c)) + 1e-10

    def test_known_value(self):
        # d(0, 9) = |9|_3 = 1/9
        assert abs(padic_distance(0, 9) - 1 / 9) < 1e-10

    def test_non_negative(self):
        for a, b in [(0, 1), (1, 4), (3, 12), (9, 18)]:
            assert padic_distance(a, b) >= 0.0


class TestPadicDigits:
    def test_zero_is_all_zeros(self):
        assert padic_digits(0, 3, 4) == [0, 0, 0, 0]

    def test_one_is_unit(self):
        assert padic_digits(1, 3, 4) == [1, 0, 0, 0]

    def test_ten_in_base_three(self):
        # 10 = 1 + 0*3 + 1*9 → digits [1, 0, 1, 0]
        assert padic_digits(10, 3, 4) == [1, 0, 1, 0]

    def test_negative_input_uses_absolute_value(self):
        assert padic_digits(-10, 3, 4) == padic_digits(10, 3, 4)

    def test_output_length_matches_n_digits(self):
        for n in [1, 3, 5, 8]:
            assert len(padic_digits(100, 3, n)) == n

    def test_base_two(self):
        # 6 in binary: 6 = 0 + 1*2 + 1*4 → [0, 1, 1, 0]
        assert padic_digits(6, 2, 4) == [0, 1, 1, 0]


class TestGoldilocksScore:
    def test_at_center_is_one(self):
        score = compute_goldilocks_score(0.5, center=0.5, width=0.15)
        assert abs(score - 1.0) < 1e-10

    def test_far_from_center_near_zero(self):
        score = compute_goldilocks_score(0.0, center=0.5, width=0.05)
        assert score < 0.001

    def test_score_bounded_between_zero_and_one(self):
        for d in [0.0, 0.25, 0.5, 0.75, 1.0]:
            s = compute_goldilocks_score(d)
            assert 0.0 <= s <= 1.0

    def test_symmetric_around_center(self):
        center = 0.5
        d1 = compute_goldilocks_score(center - 0.1, center=center)
        d2 = compute_goldilocks_score(center + 0.1, center=center)
        assert abs(d1 - d2) < 1e-10

    def test_large_distance_normalized(self):
        # With normalize=True, distance > 1 is squashed before scoring
        score_large = compute_goldilocks_score(10.0, normalize=True)
        score_near = compute_goldilocks_score(0.0, normalize=False)
        assert isinstance(score_large, float)
        assert isinstance(score_near, float)


class TestIsInGoldilocksZone:
    def test_at_center_is_in_zone(self):
        assert is_in_goldilocks_zone(0.5, center=0.5, width=0.15, threshold=0.5)

    def test_far_away_not_in_zone(self):
        assert not is_in_goldilocks_zone(0.0, center=0.5, width=0.15, threshold=0.5)

    def test_threshold_at_zero_always_true(self):
        assert is_in_goldilocks_zone(0.0, center=0.5, width=0.15, threshold=0.0)

    def test_threshold_at_one_only_at_center(self):
        # Score at center is exactly 1.0, so threshold=0.99 should pass
        assert is_in_goldilocks_zone(0.5, center=0.5, width=0.15, threshold=0.99)
        # Away from center, score < 1.0, so threshold=0.99 should fail
        assert not is_in_goldilocks_zone(0.3, center=0.5, width=0.15, threshold=0.99)


class TestPadicShift:
    def test_returns_pad_shift_result(self):
        result = padic_shift(9, shift_amount=0)
        assert isinstance(result, PAdicShiftResult)

    def test_zero_shift_identity(self):
        result = padic_shift(9, shift_amount=0)
        assert result.shift_value == 9.0

    def test_right_shift_divides(self):
        result = padic_shift(9, shift_amount=1)
        assert result.shift_value == 3.0  # 9 // 3 = 3

    def test_left_shift_multiplies(self):
        result = padic_shift(3, shift_amount=-1)
        assert result.shift_value == 9.0  # 3 * 3 = 9

    def test_canonical_form_is_string(self):
        result = padic_shift(9)
        assert isinstance(result.canonical_form, str)
        assert "3" in result.canonical_form

    def test_digits_are_list(self):
        result = padic_shift(9)
        assert isinstance(result.digits, list)
        assert len(result.digits) == 4  # default n_digits=4

    def test_valuation_is_int(self):
        result = padic_shift(9, shift_amount=0)
        assert isinstance(result.valuation, int)

    @pytest.mark.parametrize("value,shift,expected", [(27, 1, 9.0), (27, 2, 3.0), (1, -2, 9.0)])
    def test_shift_values(self, value, shift, expected):
        result = padic_shift(value, shift_amount=shift)
        assert result.shift_value == expected
