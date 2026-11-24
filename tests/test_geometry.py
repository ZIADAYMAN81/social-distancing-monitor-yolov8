import pytest

from social_distancing.geometry import Person, analyse, pair_threshold, safe_people


def person(cx, bottom, height=100, width=40):
    return Person(cx - width // 2, bottom - height, cx + width // 2, bottom)


def test_ground_point_is_bottom_centre():
    assert Person(10, 20, 50, 120).ground_point == (30, 120)


def test_fixed_pixel_mode_flags_only_the_close_pair():
    people = [person(100, 300), person(150, 300), person(600, 300)]
    pairs, violators = analyse(people, mode="pixels", min_distance_px=100)
    assert violators == {0, 1}
    assert sum(p.too_close for p in pairs) == 1


def test_people_are_either_safe_or_too_close_never_both():
    people = [person(100, 300), person(150, 300), person(600, 300)]
    _, violators = analyse(people, mode="pixels", min_distance_px=100)
    safe = safe_people(len(people), violators)
    assert safe == {2}
    assert safe.isdisjoint(violators)


def test_relative_mode_is_perspective_aware():
    # The same 100 px gap: distant (small) people are far apart in reality, near (large) people are close.
    far = [person(100, 200, height=50), person(200, 200, height=50)]
    near = [person(100, 600, height=200), person(200, 600, height=200)]
    assert analyse(far, mode="relative", height_ratio=1.2)[1] == set()
    assert analyse(near, mode="relative", height_ratio=1.2)[1] == {0, 1}


def test_threshold_values_and_invalid_mode():
    a, b = person(0, 100, height=100), person(0, 100, height=200)
    assert pair_threshold(a, b, "relative", 0, 1.0) == pytest.approx(150)
    assert pair_threshold(a, b, "pixels", 80, 1.0) == 80
    with pytest.raises(ValueError):
        pair_threshold(a, b, "metres", 80, 1.0)


def test_zero_or_one_person_has_no_pairs():
    assert analyse([]) == ([], set())
    assert analyse([person(0, 100)]) == ([], set())
