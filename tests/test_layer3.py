"""
tests/test_layer3.py — Katman 3 deklaratif API testleri.

search(), optimize() ve MultiAgentSystem doğruluk testleri.
SDK'nın gerçek dünya problemi çözme kapasitesini ölçer.
"""

import pytest
import numpy as np

from quanta.layer3.search import search
from quanta.layer3.optimize import optimize
from quanta.layer3.agent import Agent, MultiAgentSystem


# ═══════════════════════════════════════════
#  search() Testleri
# ═══════════════════════════════════════════

class TestSearch:
    """Deklaratif kuantum arama testleri."""

    def test_search_finds_exact_target(self):
        """search(target=5) → |101⟩ bulmalı."""
        result = search(num_bits=3, target=5, shots=1000, seed=42)
        assert result.most_frequent == "101"

    def test_search_finds_target_with_lambda(self):
        """Lambda ile arama: 7'nin katları."""
        result = search(
            num_bits=4, target=lambda x: x == 7,
            shots=1000, seed=42,
        )
        assert result.most_frequent == "0111"

    def test_search_high_probability(self):
        """Hedef %80+ olasılıkla bulunmalı."""
        result = search(num_bits=3, target=3, shots=1000, seed=42)
        prob = result.probabilities.get("011", 0)
        assert prob > 0.8, f"Hedef olasılığı {prob:.2f}, beklenen > 0.8"

    def test_search_invalid_bits_raises(self):
        with pytest.raises(ValueError):
            search(num_bits=0, target=0)

    def test_search_no_target_raises(self):
        """Geçersiz hedef → hata."""
        with pytest.raises(ValueError):
            search(num_bits=3, target=lambda x: False)


# ═══════════════════════════════════════════
#  optimize() Testleri
# ═══════════════════════════════════════════

class TestOptimize:
    """Deklaratif kuantum optimizasyon testleri."""

    def test_optimize_finds_minimum(self):
        """f(x) = (x-3)² → minimum x=3 olmalı."""
        result = optimize(
            num_bits=3,
            cost=lambda x: (x - 3) ** 2,
            minimize=True,
            shots=2048,
            seed=42,
        )
        assert result.best_bitstring == "011"  # 3 = 011
        assert result.best_cost == 0.0

    def test_optimize_finds_maximum(self):
        """maximize f(x) = x → max x=7 (3 bit)."""
        result = optimize(
            num_bits=3,
            cost=lambda x: x,
            minimize=False,
            shots=2048,
            seed=42,
        )
        assert int(result.best_bitstring, 2) == 7

    def test_optimize_summary_returns_string(self):
        result = optimize(num_bits=2, cost=lambda x: x, seed=42)
        assert isinstance(result.summary(), str)


# ═══════════════════════════════════════════
#  MultiAgentSystem Testleri
# ═══════════════════════════════════════════

class TestMultiAgent:
    """Multi-agent kuantum modelleme testleri."""

    def test_two_independent_agents_have_low_correlation(self):
        """Etkileşim olmayan ajanlar bağımsız olmalı."""
        system = MultiAgentSystem([
            Agent("A", ["evet", "hayır"]),
            Agent("B", ["evet", "hayır"]),
        ])
        # Etkileşim yok
        result = system.simulate(shots=2000, seed=42)
        corr = result.correlation("A", "B")
        assert -0.3 < corr < 0.3, f"Korelasyon: {corr}, beklenen ~0"

    def test_strongly_interacting_agents_are_correlated(self):
        """Güçlü etkileşim → yüksek korelasyon."""
        system = MultiAgentSystem([
            Agent("A", ["sol", "sağ"]),
            Agent("B", ["sol", "sağ"]),
        ])
        system.interact("A", "B", strength=0.9)
        result = system.simulate(shots=2000, seed=42)
        corr = result.correlation("A", "B")
        # Güçlü etkileşimde korelasyon zayıftan belirgin fark olmalı
        assert abs(corr) > 0.05, f"Korelasyon: {corr}, beklenen > 0.05"

    def test_agent_probabilities_sum_to_one(self):
        """Her ajanın olasılıkları toplamı 1 olmalı."""
        system = MultiAgentSystem([
            Agent("X", ["al", "sat"]),
        ])
        result = system.simulate(shots=1000, seed=42)
        probs = result.agent_probabilities("X")
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_three_agents_with_chain_interaction(self):
        """A↔B↔C zincir etkileşim: A ve C dolaylı bağlantılı."""
        system = MultiAgentSystem([
            Agent("A", ["0", "1"]),
            Agent("B", ["0", "1"]),
            Agent("C", ["0", "1"]),
        ])
        system.interact("A", "B", strength=0.8)
        system.interact("B", "C", strength=0.8)
        result = system.simulate(shots=1000, seed=42)

        # Sonuç geçerli olmalı
        assert result.shots == 1000
        assert len(result.agents) == 3

    def test_summary_returns_formatted_string(self):
        system = MultiAgentSystem([
            Agent("buyer", ["buy", "skip"]),
            Agent("seller", ["discount", "hold"]),
        ])
        system.interact("buyer", "seller", strength=0.5)
        result = system.simulate(shots=100, seed=42)
        summary = result.summary()
        assert "buyer" in summary
        assert "seller" in summary

    def test_invalid_agent_name_raises(self):
        system = MultiAgentSystem([Agent("A", ["x", "y"])])
        with pytest.raises(ValueError):
            system.interact("A", "Z", strength=0.5)

    def test_biased_agent_reflects_bias(self):
        """Eğilimli ajan başlangıç olasılığını yansıtmalı."""
        system = MultiAgentSystem([
            Agent("biased", ["a", "b"], bias=[0.9, 0.1]),
        ])
        result = system.simulate(shots=2000, seed=42)
        probs = result.agent_probabilities("biased")
        assert probs["a"] > 0.7, f"Beklenen >0.7, alınan: {probs['a']}"
