from unittest.mock import MagicMock

import pytest

from kung_fu_agent.training import MAX_COUNT, build_training, normalize_count, normalize_level


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("iniciante", "iniciante"),
        ("Iniciante", "iniciante"),
        ("  AVANÇADO ", "avancado"),
        ("intermediário", "intermediario"),
    ],
)
def test_normalize_level_fixes_case_and_accents(raw, expected):
    assert normalize_level(raw) == expected


@pytest.mark.parametrize("raw", ["mestre", "", None, 3])
def test_normalize_level_rejects_unknown_values(raw):
    with pytest.raises(ValueError):
        normalize_level(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(1, 1), (3, 3), (3.0, 3), (MAX_COUNT, MAX_COUNT), (500, MAX_COUNT)],
)
def test_normalize_count_accepts_and_caps(raw, expected):
    assert normalize_count(raw) == expected


@pytest.mark.parametrize("raw", [0, -1, 2.5, True, "3", None])
def test_normalize_count_rejects_invalid_values(raw):
    with pytest.raises(ValueError):
        normalize_count(raw)


def test_build_training_sends_normalized_values_as_parameters():
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [{"nome": "Diu Ma"}]

    result = build_training(conn, "Iniciante", 500)

    assert result == [{"nome": "Diu Ma"}]
    sql, params = cursor.execute.call_args.args
    assert "WHERE nivel = %s" in sql
    assert params == ("iniciante", MAX_COUNT)


def test_build_training_never_reaches_the_database_with_invalid_level():
    conn = MagicMock()

    with pytest.raises(ValueError):
        build_training(conn, "mestre", 3)

    conn.cursor.assert_not_called()