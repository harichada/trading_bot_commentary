"""v-themeshock-ailab-reroute-2026-11-16: AI-lab financing reroute + contam guards.

Flags (all default False / disabled; env wins over yaml):
  THEME_CRITIC_AILAB_FINANCING_REROUTE, THEME_CRITIC_AILAB_SECOND_CHOICE_MIN_PROB,
  THEME_CRITIC_OPTICAL_INTERCONNECT_GUARD, THEME_CRITIC_CONTAM_PLAIN_OIL.
THEME_HARD_SKIP_REQUIRE_GPU is NOT touched (asserted default False).
"""
import pytest
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

from analysis.gpu_news_critic import (
    GPUNewsCritic, GPUNewsCriticModel, KeywordMatch, Stance, CriticAction,
    ailab_financing_reroute, contam_lexicon_decision, optical_interconnect_guard_hit,
    reset_gpu_news_critic,
)

FLAGS = [
    "THEME_CRITIC_AILAB_FINANCING_REROUTE", "THEME_CRITIC_AILAB_SECOND_CHOICE_MIN_PROB",
    "THEME_CRITIC_OPTICAL_INTERCONNECT_GUARD", "THEME_CRITIC_CONTAM_PLAIN_OIL",
]
LEX = ['corning', 'glw', 'nyse:glw', 'motor oil', 'oil crisis', 'crude oil', 'petroleum',
       'opec', 'wti', 'brent', 'cohr', 'cien', 'aaoi']
TRIG = GPUNewsCriticModel.AI_COMPUTE_TRIGGERS

LEAK_DD001635 = ("Ciena, Nokia, and Cisco Named Leaders in ABI Research\u2019s Open Line "
                 "Systems for Data Center Interconnect")
GOLDEN_CRUDE_1A1E = "Crude Oil Jumps 3%; Corning Shares Move Lower"
XLE_2AF0 = "Oil Near $100 Should Lift Energy. Instead XLE ETF Is Falling \u2014 Here's Why"
RUSSIAN_4C46 = "Sorry, Trump. More Russian Oil Won\u2019t Take Down Energy Prices. Here\u2019s What Will."

# earnings_chip-top distribution like the logged 11/16 misses (flat MiniLM softmax)
EC_TOP = {"none": 0.13, "other": 0.13, "ai_compute": 0.155, "memory_hbm": 0.12,
          "ai_mega_cap": 0.14, "fed_risk_off": 0.135, "earnings_chip": 0.19}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for f in FLAGS + ["THEME_HARD_SKIP_REQUIRE_GPU"]:
        monkeypatch.delenv(f, raising=False)
    monkeypatch.setenv("GPU_CRITIC_CONTAM_LEXICON_ENABLE", "1")
    monkeypatch.delenv("GPU_CRITIC_CONTAM_LEXICON", raising=False)
    reset_gpu_news_critic()
    yield
    reset_gpu_news_critic()


# ---------------------------------------------------------------- config
class TestFlagsConfig:
    def test_defaults_off(self):
        from core.config import Config
        c = Config()
        assert c.THEME_CRITIC_AILAB_FINANCING_REROUTE is False
        assert c.THEME_CRITIC_AILAB_SECOND_CHOICE_MIN_PROB == 0.0
        assert c.THEME_CRITIC_OPTICAL_INTERCONNECT_GUARD is False
        assert c.THEME_CRITIC_CONTAM_PLAIN_OIL is False
        assert c.THEME_HARD_SKIP_REQUIRE_GPU is False  # HOLD

    def test_env_on(self, monkeypatch):
        from core.config import Config
        for f in FLAGS:
            monkeypatch.setenv(f, "1")
        monkeypatch.setenv("THEME_CRITIC_AILAB_SECOND_CHOICE_MIN_PROB", "0.16")
        c = Config()
        assert c.THEME_CRITIC_AILAB_FINANCING_REROUTE is True
        assert c.THEME_CRITIC_AILAB_SECOND_CHOICE_MIN_PROB == pytest.approx(0.16)
        assert c.THEME_CRITIC_OPTICAL_INTERCONNECT_GUARD is True
        assert c.THEME_CRITIC_CONTAM_PLAIN_OIL is True
        assert c.THEME_HARD_SKIP_REQUIRE_GPU is False

    def test_env_wins_over_yaml(self, monkeypatch):
        from core.config import Config
        c = Config()
        orig_get = c.manager.get
        monkeypatch.setattr(c.manager, "get", lambda k, d=None: True if "theme_critic_" in k else orig_get(k, d))
        assert c.THEME_CRITIC_AILAB_FINANCING_REROUTE is True  # yaml on
        monkeypatch.setenv("THEME_CRITIC_AILAB_FINANCING_REROUTE", "0")
        assert c.THEME_CRITIC_AILAB_FINANCING_REROUTE is False  # env wins

    def test_bad_threshold_is_disabled(self, monkeypatch):
        from core.config import Config
        monkeypatch.setenv("THEME_CRITIC_AILAB_SECOND_CHOICE_MIN_PROB", "abc")
        assert Config().THEME_CRITIC_AILAB_SECOND_CHOICE_MIN_PROB == 0.0


# ---------------------------------------------------------------- reroute helper
class TestAilabFinancingReroute:
    @pytest.mark.parametrize("hl", [
        "Could Anthropic's IPO Top SpaceX's $75 Billion?",
        "Anthropic\u2019s S-1 Is Here. The $518 Billion Commitment Is the Real Story.",
        "Alphabet Reportedly Values Its Anthropic Stake at $124 Billion",
        "Anthropic files to go public on Nasdaq",
        "Anthropic Reportedly Sends Investors IPO Invitations Ahead Of Potential Thanksgiving Debut",
        "Broadcom lends Anthropic $42B for custom chips",
    ])
    def test_anthropic_financing_reroutes(self, hl):
        probs, d = ailab_financing_reroute(hl, EC_TOP, True)
        assert d["original_theme"] == "earnings_chip"
        assert d["rerouted_theme"] == "ai_compute"
        assert d["reroute_reason"] == "ailab_financing"
        assert d["rerouted_sub_theme"] == "anthropic"
        assert max(probs, key=probs.get) == "ai_compute"
        assert d["theme_probs_original"] == EC_TOP
        assert sorted(probs.values()) == sorted(EC_TOP.values())  # swap only

    @pytest.mark.parametrize("hl", [
        "OpenAI raises $40 billion in funding round led by SoftBank",
        "OpenAI tender offer values company at $500 billion",
        "xAI raises $6B at $50B valuation",
        "Mistral AI closes Series C funding",
    ])
    def test_other_lab_financing_reroutes_no_anthropic_subtheme(self, hl):
        probs, d = ailab_financing_reroute(hl, EC_TOP, True)
        assert d["rerouted_theme"] == "ai_compute"
        assert d["rerouted_sub_theme"] is None

    @pytest.mark.parametrize("hl", [
        "Nvidia beats on EPS, raises guidance",
        "Anthropic Reports Q3 Net Loss of $23.29 Billion ahead of IPO",
        "OpenAI quarterly results show revenue beat; investors cheer valuation",
        "Stocks rebound as oil prices sink, Anthropic releases Q3 financials",
    ])
    def test_earnings_terms_never_reroute(self, hl):
        probs, d = ailab_financing_reroute(hl, EC_TOP, True, second_choice_min_prob=0.01)
        assert probs is EC_TOP
        assert d["rerouted_theme"] == "earnings_chip"
        assert d["reroute_reason"] == "blocked_earnings_term"

    def test_lab_without_financing_does_not_reroute(self):
        _, d = ailab_financing_reroute("Broadcom Stock: The Anthropic Panic Looks Overblown", EC_TOP, True)
        assert d["rerouted_theme"] == "earnings_chip" and d["reroute_reason"] is None

    def test_word_boundary_lab(self):
        # 'coherent' must not match 'cohere'; 'xai' inside a word must not match
        _, d = ailab_financing_reroute("Coherent raises outlook on transceiver funding", EC_TOP, True)
        assert d["reroute_ailab"] is None and d["rerouted_theme"] == "earnings_chip"

    def test_non_earnings_chip_top_untouched(self):
        p = dict(EC_TOP, ai_mega_cap=0.30)
        probs, d = ailab_financing_reroute("Anthropic IPO priced", p, True)
        assert probs is p and d["original_theme"] == d["rerouted_theme"] == "ai_mega_cap"

    def test_flag_off_identity(self):
        probs, d = ailab_financing_reroute("Anthropic IPO priced", EC_TOP, False)
        assert probs is EC_TOP and d is None

    def test_second_choice(self):
        hl = "Oracle Stock Falls on Force Majeure Notice for Data Center"
        _, d0 = ailab_financing_reroute(hl, EC_TOP, True, 0.0)
        assert d0["rerouted_theme"] == "earnings_chip"
        _, d1 = ailab_financing_reroute(hl, EC_TOP, True, 0.15)
        assert d1["rerouted_theme"] == "ai_compute" and d1["reroute_reason"] == "second_choice_ai_compute"
        _, d2 = ailab_financing_reroute(hl, EC_TOP, True, 0.16)  # 0.155 < 0.16
        assert d2["rerouted_theme"] == "earnings_chip"
        # 2nd choice not ai_compute -> no reroute
        p = dict(EC_TOP, ai_mega_cap=0.17)
        _, d3 = ailab_financing_reroute(hl, p, True, 0.01)
        assert d3["rerouted_theme"] == "earnings_chip"


# ---------------------------------------------------------------- contam helpers
class TestContamDecision:
    def _legacy(self, hl, sm, risk):
        """Verbatim copy of pre-patch _apply_contam_lexicon body."""
        hll, sml = hl.lower(), sm.lower()
        t = f"{hll} {sml}"
        if not any(k in t for k in LEX):
            return risk
        if any(x in hll for x in TRIG):
            return risk
        return max(risk, 0.65)

    @pytest.mark.parametrize("hl,sm,r", [
        (LEAK_DD001635, "", 0.0), (GOLDEN_CRUDE_1A1E, "", 0.0), (XLE_2AF0, "", 0.0),
        ("Coherent (COHR) Launches AI Datacenter Optical System", "", 0.2),
        ("AI Chips Today - Boosting AI Compute Efficiency", "", 0.0),
        ("Anthropic IPO", "crude oil", 0.1), ("Nvidia Data Center", "", 0.7),
    ])
    def test_flags_off_equals_legacy(self, hl, sm, r):
        got, _ = contam_lexicon_decision(hl, sm, r, LEX, TRIG)
        assert got == self._legacy(hl, sm, r)

    def test_leak_dd001635_guard(self):
        off, why0 = contam_lexicon_decision(LEAK_DD001635, "", 0.0, LEX, TRIG)
        assert off == 0.0 and why0 == "ai_trigger_exempt"
        on, why1 = contam_lexicon_decision(LEAK_DD001635, "", 0.0, LEX, TRIG, optical_guard=True)
        assert on >= 0.65 and why1.startswith("optical_guard:")

    def test_guard_product_requires_optical_name(self):
        # Marvell optical TP (no optical vendor name) must stay exempt
        hl = "Marvell to Showcase 2nm Optical Technology for AI Data Center Infrastructure"
        assert optical_interconnect_guard_hit(hl) is None
        r, _ = contam_lexicon_decision(hl, "", 0.0, LEX, TRIG, optical_guard=True)
        assert r == 0.0

    def test_guard_word_boundary_false_friends(self):
        for hl in ["IREN Drops as Core Scientific Pulls Back; data center capacity",
                   "AI Chips Today - Boosting AI Compute Efficiency Without Extra Power"]:
            assert optical_interconnect_guard_hit(hl) is None
            r, _ = contam_lexicon_decision(hl, "", 0.0, LEX, TRIG, optical_guard=True)
            assert r == 0.0

    def test_plain_oil(self):
        for hl in (XLE_2AF0, RUSSIAN_4C46):
            assert contam_lexicon_decision(hl, "", 0.0, LEX, TRIG)[0] == 0.0
            r, why = contam_lexicon_decision(hl, "", 0.0, LEX, TRIG, plain_oil=True)
            assert r >= 0.65 and why == "plain_oil"

    @pytest.mark.parametrize("hl", [
        "Investors toil through choppy session", "Tensions boil over at FOMC",
        "Soil quality index rises", "Earnings spoil the rally",
    ])
    def test_plain_oil_false_friends(self, hl):
        r, why = contam_lexicon_decision(hl, "", 0.0, LEX, TRIG, plain_oil=True)
        assert r == 0.0 and why is None

    def test_oilfield_counts_as_oil(self):
        r, _ = contam_lexicon_decision("Oilfield services stocks slide", "", 0.0, LEX, TRIG, plain_oil=True)
        assert r >= 0.65

    def test_plain_oil_respects_ai_trigger_exemption(self):
        hl = "SLB's Next Big Growth Engine Isn't Oil \u2014 It's the AI Data Center Boom"
        r, why = contam_lexicon_decision(hl, "", 0.0, LEX, TRIG, plain_oil=True, optical_guard=True)
        assert r == 0.0 and why == "ai_trigger_exempt"

    def test_golden_crude_oil_fp_suppressed_all_combos(self):
        for g in (False, True):
            for o in (False, True):
                r, _ = contam_lexicon_decision(GOLDEN_CRUDE_1A1E, "", 0.0, LEX, TRIG, optical_guard=g, plain_oil=o)
                assert r >= 0.65


# ---------------------------------------------------------------- model path (no torch)
class _StubModel(GPUNewsCriticModel):
    def __init__(self, probs):
        super().__init__()
        self._probs = probs
        self._loaded = True

    def _lazy_load(self):
        return True

    def _embed_scores(self, text):
        return dict(self._probs), Stance.MIXED, 0.6


@dataclass
class _Item:
    headline: str
    summary: str = ""
    symbol: str = "NVDA"
    source: str = "test"
    published_time: datetime = datetime.now(timezone.utc) - timedelta(minutes=5)

    def age_sec(self):
        return 300.0


def _critic(probs):
    c = GPUNewsCritic(engine=None, use_fake=False)
    c._model = _StubModel(probs)
    return c


class TestModelPathEndToEnd:
    def test_flags_off_card_dict_identical_keys(self):
        card = _critic(EC_TOP).score_news_item(_Item("Anthropic IPO could top SpaceX"))
        d = card.to_dict()
        assert card.reroute_diag is None and card.contam_diag is None
        for k in ("original_theme", "rerouted_theme", "reroute_reason",
                  "contam_risk_pre_guard", "contam_risk_post_guard"):
            assert k not in d
        assert max(card.theme_probs, key=card.theme_probs.get) == "earnings_chip"
        assert "reroute=" not in card.rationale_short

    def test_reroute_on_card_logs_fields(self, monkeypatch):
        monkeypatch.setenv("THEME_CRITIC_AILAB_FINANCING_REROUTE", "1")
        card = _critic(EC_TOP).score_news_item(_Item("Anthropic IPO could top SpaceX"))
        d = card.to_dict()
        assert d["original_theme"] == "earnings_chip"
        assert d["rerouted_theme"] == "ai_compute"
        assert d["reroute_reason"] == "ailab_financing"
        assert d["rerouted_sub_theme"] == "anthropic"
        assert max(card.theme_probs, key=card.theme_probs.get) == "ai_compute"
        assert set(card.relevance_by_symbol) == {"NVDA", "AMD", "AVGO", "SMCI", "ARM"}
        assert "reroute=earnings_chip->ai_compute" in card.rationale_short

    def test_reroute_on_every_row_has_fields(self, monkeypatch):
        monkeypatch.setenv("THEME_CRITIC_AILAB_FINANCING_REROUTE", "1")
        card = _critic(EC_TOP).score_news_item(_Item("Nvidia beats on EPS, raises guidance"))
        d = card.to_dict()
        assert d["original_theme"] == d["rerouted_theme"] == "earnings_chip"
        assert d["reroute_reason"] == "blocked_earnings_term"

    def test_leak_dd001635_end_to_end(self, monkeypatch):
        probs = dict(EC_TOP, ai_compute=0.25)  # critic top=ai_compute
        off = _critic(probs).score_news_item(_Item(LEAK_DD001635))
        assert off.contamination_risk < 0.6 and off.action == CriticAction.PASS
        monkeypatch.setenv("THEME_CRITIC_OPTICAL_INTERCONNECT_GUARD", "1")
        on = _critic(probs).score_news_item(_Item(LEAK_DD001635))
        d = on.to_dict()
        assert on.contamination_risk >= 0.65
        assert on.action == CriticAction.WOULD_SUPPRESS_HARD_SKIP
        assert d["contam_risk_pre_guard"] == pytest.approx(off.contamination_risk)
        assert d["contam_risk_post_guard"] >= 0.65
        assert d["contam_guard_reason"].startswith("optical_guard:")

    def test_golden_crude_end_to_end_all_flags(self, monkeypatch):
        for f in ("THEME_CRITIC_AILAB_FINANCING_REROUTE", "THEME_CRITIC_OPTICAL_INTERCONNECT_GUARD",
                  "THEME_CRITIC_CONTAM_PLAIN_OIL"):
            monkeypatch.setenv(f, "1")
        card = _critic(EC_TOP).score_news_item(_Item(GOLDEN_CRUDE_1A1E))
        assert card.contamination_risk >= 0.65
        assert card.action == CriticAction.WOULD_SUPPRESS_HARD_SKIP
        assert card.to_dict()["rerouted_theme"] == "earnings_chip"  # no AI lab -> stays

    def test_fake_critic_off_path_unchanged(self):
        c = GPUNewsCritic(use_fake=True)
        card = c.score_news_item(_Item("Anthropic CEO warns about AI compute pace"))
        assert card.reroute_diag is None and "original_theme" not in card.to_dict()


# ---------------------------------------------------------------- pre-shadow follow-ups (Research gate 11/16)
class TestPreShadowFollowups:
    # 1. bare 'coherent'
    def test_coherent_ordinary_word_does_not_suppress_anthropic_tp(self):
        hl = "Anthropic lays out coherent strategy for Claude enterprise push"
        assert optical_interconnect_guard_hit(hl) is None
        r, why = contam_lexicon_decision(hl, "", 0.0, LEX, TRIG, optical_guard=True, plain_oil=True)
        assert r == 0.0 and why is None

    def test_coherent_pricing_not_guard_bumped(self):
        hl = "Coherent pricing returns to AI chip market, says Nvidia CFO"
        assert optical_interconnect_guard_hit(hl) is None
        r, why = contam_lexicon_decision(hl, "", 0.0, LEX, TRIG, optical_guard=True, plain_oil=True)
        assert r < 0.65 and (why is None or not why.startswith("optical_guard"))

    @pytest.mark.parametrize("hl", [
        "Coherent (COHR) Launches AI Datacenter Optical System Now Shipping In Volume",  # d35ac6342b80
        "Coherent Showcases Optical Innovations to Scale AI Infrastructure at ECOC 2026",  # 12ccd221717e
        "3M Jumps 6% on Expanded Amazon Data Center Deal and Earnings Beat; Coherent Edges Higher, Corning",  # 50005b1b8370
        "Coherent shares jump on AI data center transceiver demand",
        "Coherent Corp. stock climbs as AI datacenter bets intensify",
        "Coherent\u2019s stock slides after data center guidance",
    ])
    def test_coherent_vendor_in_context_still_guarded(self, hl):
        assert optical_interconnect_guard_hit(hl) is not None
        r, why = contam_lexicon_decision(hl, "", 0.0, LEX, TRIG, optical_guard=True)
        assert r >= 0.65 and why.startswith("optical_guard:")

    def test_lowercase_coherent_with_price_word_not_vendor(self):
        assert optical_interconnect_guard_hit("AI stocks rise on coherent data center plan") is None

    # 2. plain oil + AI labs / oil-rich
    def test_oil_rich_xai_not_suppressed(self):
        hl = "Oil-rich Saudi fund invests in xAI"
        r, why = contam_lexicon_decision(hl, "", 0.0, LEX, TRIG, optical_guard=True, plain_oil=True)
        assert r == 0.0
        _, d = ailab_financing_reroute(hl, EC_TOP, True)
        assert d["rerouted_theme"] == "ai_compute"

    def test_oil_rich_alone_is_not_oil(self):
        r, why = contam_lexicon_decision("Oil-rich Norway fund boosts tech holdings", "", 0.0, LEX, TRIG, plain_oil=True)
        assert r == 0.0 and why is None

    @pytest.mark.parametrize("hl", [
        "Oil prices spike; xAI shrugs off energy costs",
        "Perplexity CEO says oil shock won't slow AI spend",
        "Mistral AI unfazed as oil jumps",
    ])
    def test_plain_oil_ai_lab_exempt(self, hl):
        r, why = contam_lexicon_decision(hl, "", 0.0, LEX, TRIG, plain_oil=True)
        assert r == 0.0 and why == "ai_lab_exempt"

    def test_mistral_wind_still_oil_contam(self):
        r, why = contam_lexicon_decision("Mistral winds batter southern France; oil tanker delayed",
                                         "", 0.0, LEX, TRIG, plain_oil=True)
        assert r >= 0.65 and why == "plain_oil"

    def test_legacy_lexicon_with_ai_lab_unchanged(self):
        # exemption is plain-oil only; legacy crude-oil hit with xAI still suppresses (as before)
        hl = "Crude oil jumps; xAI shares quiet"
        assert contam_lexicon_decision(hl, "", 0.0, LEX, TRIG)[0] >= 0.65
        assert contam_lexicon_decision(hl, "", 0.0, LEX, TRIG, optical_guard=True, plain_oil=True)[0] >= 0.65

    def test_xle_russian_oil_still_fixed(self):
        for hl in (XLE_2AF0, RUSSIAN_4C46):
            assert contam_lexicon_decision(hl, "", 0.0, LEX, TRIG, plain_oil=True)[0] >= 0.65

    # 3. top-2 margin logged on every row
    def test_top2_margin_every_row(self):
        for hl in ("Anthropic IPO priced", "Nvidia beats on EPS, raises guidance", "Unrelated headline"):
            _, d = ailab_financing_reroute(hl, EC_TOP, True)
            assert d["theme_second"] == "ai_compute"
            assert d["theme_top2_margin"] == pytest.approx(0.19 - 0.155)
        _, d = ailab_financing_reroute("x", {"none": 1.0}, True)
        assert d["theme_top2_margin"] is None and d["theme_second"] is None

    def test_top2_margin_in_card_dict(self, monkeypatch):
        monkeypatch.setenv("THEME_CRITIC_AILAB_FINANCING_REROUTE", "1")
        d = _critic(EC_TOP).score_news_item(_Item("Anthropic IPO could top SpaceX")).to_dict()
        assert d["theme_top2_margin"] == pytest.approx(0.035)

    def test_top2_margin_absent_flag_off(self):
        d = _critic(EC_TOP).score_news_item(_Item("Anthropic IPO could top SpaceX")).to_dict()
        assert "theme_top2_margin" not in d
