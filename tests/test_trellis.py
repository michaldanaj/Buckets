# coding: utf-8
"""Smoke testy wykresów Trellis (spec/raport-w-czasie.md, sekcja 3.2)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import buckets.trellis as tr
from buckets.over_time import DistributionOverTime


def make_dot(n=400, seed=3, pred=True, n_buckets=5):
    rng = np.random.default_rng(seed)
    buckets = [f"b{i}" for i in range(n_buckets)]
    df = pd.DataFrame({
        "bucket": rng.choice(buckets, size=n),
        "czas": rng.choice([f"2024-{m:02d}" for m in range(1, 13)], size=n),
    })
    df["target"] = (rng.random(n) < 0.3).astype(int)
    df["pred"] = rng.random(n)
    return DistributionOverTime(
        df["czas"], df["bucket"], df["target"],
        pred=df["pred"] if pred else None,
        var_order=buckets,
    )


@pytest.fixture(autouse=True)
def zamykaj_figury():
    yield
    plt.close("all")


class TestSmoke:
    def test_plot_distribution(self):
        dot = make_dot()
        fig = tr.plot_distribution(dot, "zmienna")
        assert isinstance(fig, plt.Figure)
        # liczba paneli = liczba bucketów
        assert len([ax for ax in fig.axes if ax.get_title()]) == 5

    def test_plot_avg_target_by_bucket(self):
        dot = make_dot()
        fig = tr.plot_avg_target_by_bucket(dot, "zmienna")
        assert isinstance(fig, plt.Figure)
        assert len([ax for ax in fig.axes if ax.get_title()]) == 5

    def test_plot_avg_target_by_period(self):
        dot = make_dot()
        fig = tr.plot_avg_target_by_period(dot, "zmienna")
        assert isinstance(fig, plt.Figure)
        # panel na okres
        assert len([ax for ax in fig.axes if ax.get_title()]) == 12

    def test_plot_pit_ttc(self):
        fig = tr.plot_pit_ttc(make_dot(), "zmienna")
        assert isinstance(fig, plt.Figure)
        # dwie serie: observed + estimated
        assert len(fig.axes[0].lines) == 2

    def test_plot_pit_ttc_bez_pred(self):
        fig = tr.plot_pit_ttc(make_dot(pred=False), "zmienna")
        assert len(fig.axes[0].lines) == 1

    def test_jeden_bucket_degeneracja(self):
        dot = make_dot(n_buckets=1)
        fig = tr.plot_distribution(dot, "zmienna")
        assert isinstance(fig, plt.Figure)

    def test_kolejnosc_paneli_wg_var_order(self):
        dot = make_dot()
        fig = tr.plot_avg_target_by_bucket(dot, "zmienna")
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert titles == ["b0", "b1", "b2", "b3", "b4"]
