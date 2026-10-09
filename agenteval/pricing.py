"""Configurable per-million-token pricing used to cost a run.

Prices are deliberately not baked into any agent: a run takes a ``PriceTable``
so the same trace can be re-costed when a vendor changes list prices. The
bundled defaults are Anthropic's published list prices (USD per million tokens,
standard tier) and nothing in the harness depends on them.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DEFAULT_PRICES_PER_MTOK", "PriceTable"]

#: model-id prefix -> (input USD/MTok, output USD/MTok)
DEFAULT_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-4-5": (5.0, 25.0),
    "claude-opus-4-1": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-3-5": (0.80, 4.0),
}


@dataclass(frozen=True)
class PriceTable:
    """Input/output price per million tokens, plus where the numbers came from."""

    input_per_mtok: float
    output_per_mtok: float
    source: str = "explicit"

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Return the USD cost of a token count."""
        return (
            input_tokens / 1_000_000 * self.input_per_mtok
            + output_tokens / 1_000_000 * self.output_per_mtok
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict for the run artifact."""
        return {
            "input_per_mtok": self.input_per_mtok,
            "output_per_mtok": self.output_per_mtok,
            "source": self.source,
        }

    @classmethod
    def for_model(
        cls,
        model: str,
        *,
        input_price: float | None = None,
        output_price: float | None = None,
    ) -> PriceTable:
        """Resolve prices from explicit flags, then the bundled table, then zero.

        An unpriced model costs 0 rather than a guessed number, so a report
        never quietly invents a price for a model we do not know.
        """
        if input_price is not None or output_price is not None:
            found = cls._lookup(model)
            fallback_in, fallback_out = found if found else (0.0, 0.0)
            return cls(
                input_per_mtok=input_price if input_price is not None else fallback_in,
                output_per_mtok=output_price if output_price is not None else fallback_out,
                source="cli-override",
            )
        found = cls._lookup(model)
        if found:
            return cls(found[0], found[1], source="bundled-table")
        return cls(0.0, 0.0, source="unpriced-model")

    @staticmethod
    def _lookup(model: str) -> tuple[float, float] | None:
        """Return the price pair for the longest matching model prefix."""
        name = (model or "").lower()
        candidates = [
            (prefix, prices)
            for prefix, prices in DEFAULT_PRICES_PER_MTOK.items()
            if name.startswith(prefix) or prefix in name
        ]
        if not candidates:
            return None
        prefix, prices = max(candidates, key=lambda item: len(item[0]))
        return prices
